#!/usr/bin/env bash
set -ex
cd toolkits/model_checkpoints_convertor/FineRMoE/ckpt_convert_utils/

export CUDA_VISIBLE_DEVICES=0
START_TIME=$SECONDS
MASTER_ADDR=localhost
MASTER_PORT=$(shuf -n 1 -i 10000-65535)

########################
# PARSE HYPER-PARAMETERS
########################
MCORE_PATH=${1}
HF_PATH=$2
HF_ARCH_PATH=$3
MG2HF=$4
MODEL_SIZE=${5:-'A7B'}
PR=bf16

MODEL_NAME=$(basename "${MCORE_PATH}")
MODEL_NAME="${MODEL_NAME%%-DataMix*}"  

TP="${MODEL_NAME#*TP}"  
TP="${TP%%-*}"  
PP="${MODEL_NAME#*PP}"  
PP="${PP%%-*}"  
ETP=1
EP="${MODEL_NAME#*EP}"  
EP="${EP%%-*}"  

if [[ "$MODEL_NAME" == *G_I* ]]; then
    G_I=$(echo "$MODEL_NAME" | grep -oP '(?<=G_I)\d+')
else
    G_I=1
fi

if [[ "$MODEL_NAME" == *G_O* ]]; then
    G_O=$(echo "$MODEL_NAME" | grep -oP '(?<=G_O)\d+')
else
    G_O=1
fi

if [[ "$MODEL_NAME" == *R_I* ]]; then
    R_I=$(echo "$MODEL_NAME" | grep -oP '(?<=R_I)\d+')
else
    R_I=0
fi

if [[ "$MODEL_NAME" == *R_O* ]]; then
    R_O=$(echo "$MODEL_NAME" | grep -oP '(?<=R_O)\d+')
else
    R_O=0
fi

if [[ $MODEL_NAME =~ "ConcatProj" ]]; then
    concat_options=" \
		    --moe-concat-proj"
else
    concat_options=""
fi

CURRENT_DIR="$( cd "$( dirname "$0" )" && pwd )"
MEGATRON_PATH=$(dirname $(dirname $(dirname $( dirname ${CURRENT_DIR}))))
export PYTHONPATH=$PYTHONPATH:${MEGATRON_PATH}:${MEGATRON_PATH}/Megatron-LM-250328

if [[ ${MODEL_SIZE} == 'A7B' ]]; then
    NUM_LAYERS=28
    HIDDEN_SIZE=3584
    NUM_ATTENTION_HEADS=28
    INTERMEDIATE_SIZE=18944
    NUM_KEY_VALUE_HEADS=4
    MAX_POSITION_EMBEDDINGS=131072
    EXTRA_VOCAB_SIZE=421
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
    RMS_NORM_EPS=1e-6
    tie_option=" \
            --untie-embeddings-and-output-weights \
            "
else
    echo "No such setting for model_size as ${MODEL_SIZE}"
    exit 1
fi

NUM_SHARED_EXPERTS=1
MOE_INTERMEDIATE_SIZE="${MODEL_NAME#*EI}"  
MOE_INTERMEDIATE_SIZE="${MOE_INTERMEDIATE_SIZE%%-*}"  
ROPE_THETA=1000000
NUM_EXPERTS="${MODEL_NAME#*NumExpert}"  
NUM_EXPERTS="${NUM_EXPERTS%%-*}"  
ROUTER_TOPK="${MODEL_NAME#*TOP}"
ROUTER_TOPK="${ROUTER_TOPK%%-*}"  
SHARED_EXPERT_INTERMEDIATE_SIZE="${MODEL_NAME#*SI}"  
SHARED_EXPERT_INTERMEDIATE_SIZE="${SHARED_EXPERT_INTERMEDIATE_SIZE%%-*}"  
moe_freq='([1]*'${NUM_LAYERS}')'
# moe_freq=\'${moe_freq}\'
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

if [ "${ROUTER_TOPK}" -eq 1 ]; then
    moe_options="${moe_options} --moe-router-pre-softmax"
fi

cpu_options=""

if [ $MG2HF = true ]; then
    convert_options=" \
                --convert-checkpoint-from-megatron-to-transformers \
                --hf-ckpt-path ${HF_ARCH_PATH}"
    SOURCE_PATH=${MCORE_PATH}
    TARGET_PATH=${HF_PATH}

elif [[ $MG2HF == false || ${MG2HF} == "upcycle" ]]; then
    convert_options=""
    SOURCE_PATH=${HF_PATH}
    TARGET_PATH=${MCORE_PATH}

elif [ $MG2HF = "convert" ]; then
    convert_options="--no_upcycle"
    SOURCE_PATH=${HF_PATH}
    TARGET_PATH=${MCORE_PATH}
fi

if [ $PR = fp16 ]; then
    pr_options=" \
		    --fp16"

elif [ $PR = bf16 ]; then
    pr_options=" \
        --bf16"

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

cd toolkits/model_checkpoints_convertor/FineRMoE
torchrun ${DISTRIBUTED_ARGS} finermoe_converter.py \
    --load ${SOURCE_PATH} \
    --save ${TARGET_PATH} \
    --target-tensor-model-parallel-size ${TP} \
    --target-pipeline-model-parallel-size ${PP} \
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
    ${concat_options} \
    ${extra_options}

ELAPSED_TIME=$(($SECONDS - $START_TIME))
echo "$(($ELAPSED_TIME/60)) min $(($ELAPSED_TIME%60)) sec"

