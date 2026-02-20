#!/usr/bin/env bash
set -ex
# Get Env Params
if [[ -z ${WORLD_SIZE} || ${WORLD_SIZE} == 0 ]]; then
    export NODE_RANK=0
elif [[ -z ${NODE_RANK} ]]; then
    export NODE_RANK=${RANK}
fi

export CUDA_VISIBLE_DEVICES=0
START_TIME=$SECONDS
MASTER_ADDR=localhost
MASTER_PORT=$(shuf -n 1 -i 10000-65535)

MODEL_SIZE=${1}
SOURCE_CKPT_PATH=${2}
T_I=${3}
TP=${4}
PP=${5}
ETP=${6}
EP=${7}
PR=${8}
SHARED_EXPERT_INIT=${9}   ## copy, noshare
ROUTED_EXPERT_INIT=${10}  ## copy, finermoe, all_normal
ROUTER_INIT=${11}         ## normal
FINE_PARAM=${12}          ## G_I-G_O-R_I-R_O
MG2HF=${13}
HF_CKPT_PATH=${14}
force_init=${15:-'false'}
concat_proj=${16:-'false'}
CONCAT_PROJ_INIT=${17:-'normal'}
BASE_PATH=${18}

IFS='/' read -ra parts <<< "$SOURCE_CKPT_PATH"
MODEL_NAME=${parts[-1]}

IFS='-' read -ra parts <<< "$FINE_PARAM"
num_parts=${#parts[@]}

G_I="${parts[0]}"
G_O="${parts[1]}"
R_I="${parts[2]}"
R_O="${parts[3]}"

N_EXPERTS=$((G_I*G_O*R_I*R_O))
ACT_EXPERTS=$((T_I*G_O))

CURRENT_DIR="$( cd "$( dirname "$0" )" && pwd )"
MEGATRON_PATH=$(dirname $(dirname $(dirname $( dirname ${CURRENT_DIR}))))
export PYTHONPATH=$PYTHONPATH:${MEGATRON_PATH}:${MEGATRON_PATH}/Megatron-LM-250328

if [ $MODEL_SIZE = 'A7B' ]; then
    NUM_LAYERS=28
    HIDDEN_SIZE=3584
    NUM_ATTENTION_HEADS=28
    INTERMEDIATE_SIZE=18944
    NUM_KEY_VALUE_HEADS=4
    MAX_POSITION_EMBEDDINGS=131072
    EXTRA_VOCAB_SIZE=421
    NUM_SHARED_EXPERTS=1
    RMS_NORM_EPS=1e-6
    tie_option=" \
            --untie-embeddings-and-output-weights \
            "
elif [[ ${MODEL_SIZE} == 'UpFrom0.5B' ]]; then
    NUM_LAYERS=24
    HIDDEN_SIZE=896
    NUM_ATTENTION_HEADS=14
    INTERMEDIATE_SIZE=4864
    NUM_KEY_VALUE_HEADS=2
    MAX_POSITION_EMBEDDINGS=32768
    EXTRA_VOCAB_SIZE=293
    NUM_SHARED_EXPERTS=1
    RMS_NORM_EPS=1e-6
    tie_option=" \
            --untie-embeddings-and-output-weights \
            "
elif [[ ${MODEL_SIZE} == 'UpFrom1.5B' ]]; then
    NUM_LAYERS=28
    HIDDEN_SIZE=1536
    NUM_ATTENTION_HEADS=12
    INTERMEDIATE_SIZE=8960
    NUM_KEY_VALUE_HEADS=2
    MAX_POSITION_EMBEDDINGS=32768
    EXTRA_VOCAB_SIZE=293
    NUM_SHARED_EXPERTS=1
    RMS_NORM_EPS=1e-6
    tie_option=" \
            --untie-embeddings-and-output-weights \
            "
else
    echo "No such setting for model_size as ${MODEL_SIZE}"
    exit 1
fi

if [[ ${SHARED_EXPERT_INIT} == 'noshare' ]]; then
    SHARE_EXPERT_MOE_INTER_SIZE=0
else
    SHARE_EXPERT_MOE_INTER_SIZE=$((INTERMEDIATE_SIZE / 1))
fi

SPARSE_EXPERT_MOE_INTER_SIZE=$((INTERMEDIATE_SIZE / G_I))

if [ -n "$SHARE_EXPERT_MOE_INTER_SIZE" ] && [ "$SHARE_EXPERT_MOE_INTER_SIZE" -gt 0 ]; then
    NUM_SHARE=1
else
    NUM_SHARE=0
fi

TARGET_CKPT_PATH=${BASE_PATH}/hf2mcore/${MODEL_NAME}-${NUM_SHARE}S-NumExpert${N_EXPERTS}-TP${TP}-PP${PP}-EP${EP}-SI${SHARE_EXPERT_MOE_INTER_SIZE}-EI${SPARSE_EXPERT_MOE_INTER_SIZE}-TOP${ACT_EXPERTS}-G_I${G_I}-G_O${G_O}-SEI_${SHARED_EXPERT_INIT}-REI_${ROUTED_EXPERT_INIT}-RI_${ROUTER_INIT}-R_I${R_I}-R_O${R_O}

if [ ${concat_proj} = true ]; then
    TARGET_CKPT_PATH=${TARGET_CKPT_PATH}-ConcatProj
fi

echo "export MCORE_PATH=${TARGET_CKPT_PATH}" > /tmp/env_vars.sh

if [ ${force_init} = true ]; then
    rm -rf ${TARGET_CKPT_PATH}
fi

if [[ (-d "${TARGET_CKPT_PATH}" && -d "${TARGET_CKPT_PATH}/release") || "${NODE_RANK}" -gt 0 ]]; then
    echo "${TARGET_CKPT_PATH} has existed"
    exit 0
else
    echo ${TARGET_CKPT_PATH}
fi

MOE_INTERMEDIATE_SIZE=${SPARSE_EXPERT_MOE_INTER_SIZE}
ROPE_THETA=1000000
NUM_EXPERTS=${N_EXPERTS}
ROUTER_TOPK=${ACT_EXPERTS}
SHARED_EXPERT_INTERMEDIATE_SIZE=${SHARE_EXPERT_MOE_INTER_SIZE}
moe_freq='([1]*'${NUM_LAYERS}')'
moe_options=" \
    --moe-grouped-gemm \
    --moe-token-dispatcher-type alltoall \
    --moe-router-topk ${ROUTER_TOPK} \
    --num-experts ${NUM_EXPERTS} \
    --target-expert-tensor-parallel-size ${ETP} \
    --target-expert-model-parallel-size ${EP} \
    --moe-ffn-hidden-size ${MOE_INTERMEDIATE_SIZE} \
    --moe-router-load-balancing-type aux_loss \
    --moe-aux-loss-coeff 0.001 \
    --moe-layer-freq ${moe_freq} \
    "

if [ -n "$SHARED_EXPERT_INTERMEDIATE_SIZE" ] && [ "$SHARED_EXPERT_INTERMEDIATE_SIZE" -gt 0 ]; then
    moe_options="${moe_options} --moe-shared-expert-intermediate-size ${SHARED_EXPERT_INTERMEDIATE_SIZE}"
fi

cpu_options=""
# cpu_options=" \
#                 --use-cpu-initialization"

if [ "${ROUTER_TOPK}" -eq 1 ]; then
    moe_options="${moe_options} --moe-router-pre-softmax"
fi

if [ $MG2HF = true ]; then
    convert_options=" \
                --convert-checkpoint-from-megatron-to-transformers \
                --hf-ckpt-path ${HF_CKPT_PATH}"

elif [ $MG2HF = false ]; then
    convert_options=""
fi

if [ $PR = fp16 ]; then
    pr_options=" \
		    --fp16"

elif [ $PR = bf16 ]; then
    pr_options=" \
        --bf16"

fi

if [ ${concat_proj} = true ]; then
    concat_options=" \
		    --moe-concat-proj \
            --concatproj_init_method ${CONCAT_PROJ_INIT}"
else
    concat_options=""
fi


if [ -z ${MP_PP0_LAYERS} ];then
    uneven_split_option=""
elif [ ${PP} -gt 1 ]; then
    _check=$(( ( $NUM_LAYERS - ${MP_PP0_LAYERS} ) % ( ${PP} - 1 ) ))
    if [ $_check != 0 ]; then
        echo "With uneven pipelineing the left over layers must be divisible by left over stages."
        exit -1
    fi

    uneven_split_option=" \
        --target-decoder-first-pipeline-num-layers ${MP_PP0_LAYERS}
    "
else
    echo "uneven pipeline split must be used when PP > 1"
    exit -1
fi


DISTRIBUTED_ARGS="--nproc_per_node 1 --nnodes 1 --node_rank 0 --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

cd toolkits/model_checkpoints_convertor/FineRMoE/
torchrun ${DISTRIBUTED_ARGS} finermoe_converter.py \
    --load ${SOURCE_CKPT_PATH} \
    --save ${TARGET_CKPT_PATH} \
    --target-tensor-model-parallel-size ${TP} \
    --target-pipeline-model-parallel-size ${PP} \
    --select_shared_expert_method ${SHARED_EXPERT_INIT} \
    --expert_init_method ${ROUTED_EXPERT_INIT} \
    --router_init_method ${ROUTER_INIT} \
    --micro-batch-size 1 \
    --save-interval 1 \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --swiglu \
    --num-layers ${NUM_LAYERS} \
    --hidden-size ${HIDDEN_SIZE} \
    --ffn-hidden-size ${INTERMEDIATE_SIZE} \
    --num-attention-heads ${NUM_ATTENTION_HEADS} \
    --max-position-embeddings 10 \
    --max-padding-length 10 \
    --seq-length 10 \
    --no-async-tensor-model-parallel-allreduce \
    --patch-tokenizer-type Qwen2Tokenizer \
    --extra-vocab-size ${EXTRA_VOCAB_SIZE} \
    --no-bias-swiglu-fusion \
    --no-rope-fusion \
    --position-embedding-type rope \
    --use-rotary-position-embeddings \
    --disable-bias-linear \
    --normalization RMSNorm \
    --norm-epsilon ${RMS_NORM_EPS} \
    --rotary-base ${ROPE_THETA} \
    --rotary-seq-len-interpolation-factor 1 \
    --transformer-impl transformer_engine \
    --attention-backend fused \
    --use-rope-scaling \
    --group-query-attention \
    --num-query-groups ${NUM_KEY_VALUE_HEADS} \
    --add-qkv-bias \
    --dist-ckpt-strictness ignore_all \
    --G-I ${G_I} \
    --G-O ${G_O} \
    --R-I ${R_I} \
    --R-O ${R_O} \
    ${tie_option} \
    ${moe_options} \
    ${convert_options} \
    ${pr_options} \
    ${uneven_split_option} \
    ${cpu_options} \
    ${extra_options} \
    ${concat_options}

ELAPSED_TIME=$(($SECONDS - $START_TIME))
echo "$(($ELAPSED_TIME/60)) min $(($ELAPSED_TIME%60)) sec"
