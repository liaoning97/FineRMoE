#!/bin/bash
set -e
ENV=${1}

if [ $ENV = dsw ]; then
    if [[ -z ${CUDA_VISIBLE_DEVICES} ]]; then
        export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    fi
    MASTER_ADDR=${MASTER_ADDR}
    MASTER_PORT=${MASTER_PORT}
    NNODES=${WORLD_SIZE}
    NODE_RANK=${RANK}
    GPUS_PER_NODE=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
elif [ $ENV = dlc ]; then
    NNODES=${WORLD_SIZE}
    NODE_RANK=${RANK}
    GPUS_PER_NODE=${KUBERNETES_CONTAINER_RESOURCE_GPU}
fi
if [[ -z ${NNODES} ]]; then
    NNODES=1
    NODE_RANK=0
    MASTER_ADDR=localhost
    MASTER_PORT=29500
fi

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

### BASE CONFIG ###
MODEL_SIZE=${2}
BATCH_SIZE=${3}
GLOBAL_BATCH_SIZE=${4}
LR=${5}
MIN_LR=${6}
SEQ_LEN=${7}
PAD_LEN=${8}
PR=${9}
### BASE CONFIG ###

### PARALLEL / BOOL OPTION ###
TP=${10}
PP=${11}
CP=${12}
ETP=${13}
EP=${14}
SP=${15}
DO=${16}
FL=${17}
SFT=${18}
### PARALLEL / BOOL OPTION ###

### OTHERS ###
AC=${19}
OPTIMIZER_OFFLOAD=${20}
SAVE_INTERVAL=${21}
DATASET_PATH=${22}
VALID_DATASET_PATH=${23}
PRETRAIN_CHECKPOINT_PATH=${24}

# the following two values will not be used when SFT is true
TRAIN_TOKENS=${25}
WARMUP_TOKENS=${26}
###############################

OUTPUT_BASEPATH=${27}
FREEZE_BACKBONE=${28}
RESUME=${29:-'true'}

CURRENT_DIR="$( cd "$( dirname "$0" )" && pwd )"
MEGATRON_PATH=$( dirname $( dirname ${CURRENT_DIR}))
export PYTHONPATH=${MEGATRON_PATH}:${MEGATRON_PATH}/Megatron-LM-250328:$PYTHONPATH
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=true # for PyTorch >= 2.6

## ---------mix weight in DATASET_PATH---
## input format: 
## 1. file
## 2. file1 file2 ...
## 3. dir/
## 4. dir/::keyword1:weight1-keyword2:weight2

parse_dataset_path() {
  local input="$1"
  local output=()

  if [[ "$input" == *"::"* ]]; then
    # Case 4: path::kwd1:weight1-kwd2:weight2...
    local path="${input%%::*}"
    local spec="${input##*::}"

    IFS='-' read -ra pairs <<< "$spec"
    for pair in "${pairs[@]}"; do
      kwd="${pair%%:*}"
      weight="${pair##*:}"

      while IFS= read -r file; do
        filename=$(basename "$file")
        if [[ "${filename,,}" == *"${kwd,,}"* && "${filename,,}" == *.bin ]]; then
          file_no_ext="${file%.bin}"
          output+=("$weight" "$file_no_ext")
        fi
      done < <(find "$path" -type f -iname "*.bin" -o -type l -iname "*.bin")
#      done < <(find "$path" -type f \( -iname "*.bin" \))
      

    done

  elif [[ -d "$input" ]]; then
    # Case 3: a directory
    while IFS= read -r file; do
      file_no_ext="${file%.bin}"
      output+=("$file_no_ext")
    done < <(find "$input" -maxdepth 1 -type f \( -iname "*.bin" \))

  else
    # Case 1 & 2: one or more file path prefixes (space separated)
    IFS=' ' read -ra paths <<< "$input"
    for prefix in "${paths[@]}"; do
      candidate="${prefix}.bin"
      if [[ -f "$candidate" ]]; then
        output+=("$prefix")
      else
        echo "Invalid prefix: $prefix (no matching .bin file)" >&2
        exit 1
      fi
    done
  fi

  if [[ ${#output[@]} -eq 0 ]]; then
    # 如果没有成功匹配，保持原始输入
    echo "$input"
  else
    echo "${output[@]}"
  fi
#   echo "${output[@]}"
}

if [[ "$DATASET_PATH" == *"::"* ]]; then
    spec="${DATASET_PATH##*::}"
elif [[ -d "$DATASET_PATH" ]]; then
    IFS='/' read -ra parts <<< "$DATASET_PATH"
    spec=${parts[-1]}
else
    spec=${DATASET_PATH}
fi
DATASET_PATH=$(parse_dataset_path "$DATASET_PATH")
echo "DATASET_PATH = ${DATASET_PATH}"


## ---------mix weight in DATASET_PATH---
### OTHERS ###

if [ $FL = true ]; then
    export NVTE_FLASH_ATTN=1 NVTE_FUSED_ATTN=0
    attn_backend_option=" \
        --attention-backend flash
    "
elif [ $FL = false ]; then
    export NVTE_FLASH_ATTN=0 NVTE_FUSED_ATTN=1
    attn_backend_option=" \
        --attention-backend fused
    "
fi

IFS='/' read -ra parts <<< "$PRETRAIN_CHECKPOINT_PATH"
MODEL_NAME=${parts[-1]}

if [[ "$MODEL_NAME" == *Data* ]]; then
    MODEL_NAME="${MODEL_NAME%-Data*}"
fi

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

if [ $MODEL_SIZE = 'A7B' ]; then
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

MOE_INTERMEDIATE_SIZE="${MODEL_NAME#*EI}"  
MOE_INTERMEDIATE_SIZE="${MOE_INTERMEDIATE_SIZE%%-*}"  
ROPE_THETA=1000000
NUM_EXPERTS="${MODEL_NAME#*NumExpert}"  
NUM_EXPERTS="${NUM_EXPERTS%%-*}"  
ROUTER_TOPK="${MODEL_NAME#*TOP}"
ROUTER_TOPK="${ROUTER_TOPK%%-*}"  
NUM_SHARED_EXPERTS=1
SHARED_EXPERT_INTERMEDIATE_SIZE="${MODEL_NAME#*SI}"  
SHARED_EXPERT_INTERMEDIATE_SIZE="${SHARED_EXPERT_INTERMEDIATE_SIZE%%-*}"  
moe_freq='([1]*'${NUM_LAYERS}')'
moe_freq=\'${moe_freq}\'

moe_options=" \
    --moe-grouped-gemm \
    --moe-token-dispatcher-type alltoall \
    --moe-router-topk ${ROUTER_TOPK} \
    --num-experts ${NUM_EXPERTS} \
    --expert-tensor-parallel-size ${ETP} \
    --expert-model-parallel-size ${EP} \
    --moe-ffn-hidden-size ${MOE_INTERMEDIATE_SIZE} \
    --moe-router-load-balancing-type seq_aux_loss \
    --moe-aux-loss-coeff 0.001 \
    --moe-layer-freq ${moe_freq} \
    --moe-layer-recompute \
    --moe-shared-expert-overlap \
    "

if [ $FREEZE_BACKBONE = true ]; then
    moe_options="${moe_options} --only-train-list embedding+router+mlp.experts+layernorm"
    echo "Freeze Backbone."
fi

if [ -n "$SHARED_EXPERT_INTERMEDIATE_SIZE" ] && [ "$SHARED_EXPERT_INTERMEDIATE_SIZE" -gt 0 ]; then
    moe_options="${moe_options} --moe-shared-expert-intermediate-size ${SHARED_EXPERT_INTERMEDIATE_SIZE}"
fi

if [ "${ROUTER_TOPK}" -eq 1 ]; then
    moe_options="${moe_options} --moe-router-pre-softmax"
fi

new_spec=${spec//:/_}

TRAIN_TOKENS_B=$(awk "BEGIN {print $TRAIN_TOKENS / 1000000000}")
WARMUP_TOKENS_B=$(awk "BEGIN {print $WARMUP_TOKENS / 1000000000}")

NAME=${MODEL_NAME}-Data${new_spec}-Train${TRAIN_TOKENS_B}B-WMUP${WARMUP_TOKENS_B}B-LR${LR}-MIN_LR${MIN_LR}
if [ $FREEZE_BACKBONE = true ]; then
    NAME=${MODEL_NAME}-Data${new_spec}-Train${TRAIN_TOKENS_B}B-WMUP${WARMUP_TOKENS_B}B-LR${LR}-MIN_LR${MIN_LR}-FreezeBackbone 
fi

# Here are some configs controled by env
if [ -z ${MP_DATASET_TYPE} ];then
    MP_DATASET_TYPE="idxmap"
fi

if [ -z ${MP_AC_LAYERS} ];then
    MP_AC_LAYERS=1
fi

if [ -z ${MP_VP} ]; then
    vp_option=""
else
    vp_option=" \
        --num-layers-per-virtual-pipeline-stage ${MP_VP}"
fi

if [ -z ${MP_SFT_PACKING} ]; then
    MP_SFT_PACKING=false
fi

TP_COMM_OVERLAP=$(( ($TP > 1) ? 1 : 0 ))
comm_overlap_option="\
    --overlap-grad-reduce \
    --overlap-param-gather"


if [ $TP_COMM_OVERLAP -eq 1 ]; then
    comm_overlap_option="\
        --tp-comm-overlap \
        --overlap-grad-reduce \
        --overlap-param-gather"
fi

if [ $AC = full ]; then
    _check=$(( ($NUM_LAYERS / $PP) % ${MP_AC_LAYERS} ))
    if [ $_check != 0 ]; then
        echo "the num layers per pp rank must be a multiple of the recompute layers."
        exit -1
    fi
    activation_checkpoint_options=" \
		    --recompute-method uniform \
            --recompute-num-layers ${MP_AC_LAYERS} \
		    --recompute-granularity full"
elif [ $AC = sel ]; then
    activation_checkpoint_options=" \
        --recompute-activations"
elif [ $AC = none ]; then
    activation_checkpoint_options=" \
    "
elif [ $AC = offload ]; then
    activation_checkpoint_options=" \
		    --cpu-offloading \
		    --cpu-offloading-num-layers ${MP_AC_LAYERS}"
    if [ $TP_COMM_OVERLAP -eq 1 ]; then
        echo "Disable --overlap-grad-reduce and --overlap-param-gather when cpu offloading is on..."
        comm_overlap_option="\
            --tp-comm-overlap"
    else
        echo "Disable --overlap-grad-reduce and --overlap-param-gather when cpu offloading is on..."
        comm_overlap_option=""
    fi
fi

if [ $PR = fp16 ]; then
    pr_options=" \
		    --fp16 \
            --apply-query-key-layer-scaling"
    export NVTE_APPLY_QK_LAYER_SCALING=1
elif [ $PR = bf16 ]; then
    pr_options=" \
        --bf16"
elif [ $PR = fp8 ]; then
    pr_options=" \
        --bf16 \
        --fp8-format hybrid \
        --fp8-amax-compute-algo max \
        --fp8-amax-history-len 1024"
fi

if [ $OPTIMIZER_OFFLOAD != false ] && [ $DO = false ]; then
    echo "Offload optimizer is valid only if \$DO=true"
    DO=true
fi

if [ $DO = true ]; then
    do_option=" \
		    --use-distributed-optimizer"

elif [ $DO = false ]; then
    do_option=" \
                    "
fi


if [ $SP = true ] && [ $TP -gt 1 ]; then
    sp_option=" \
		    --sequence-parallel"

elif [ $SP = false ]; then
    sp_option=" \
                    "
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
        --decoder-first-pipeline-num-layers ${MP_PP0_LAYERS}
    "
else
    echo "uneven pipeline split must be used when PP > 1"
    exit -1
fi

if [ $PRETRAIN_CHECKPOINT_PATH != none ]; then
    load_option=" \
            --load $PRETRAIN_CHECKPOINT_PATH \
            --no-load-rng \
            --no-load-optim "
elif [ $PRETRAIN_CHECKPOINT_PATH = none ]; then
    load_option=" \
            --no-load-rng \
            --no-load-optim "
fi

if [ $OPTIMIZER_OFFLOAD != false ]; then
    offload_option=" \
        --optimizer-cpu-offload \
        --use-precision-aware-optimizer \
        --optimizer-offload-fraction ${OPTIMIZER_OFFLOAD}"
fi

if [ $SFT = true ]; then
    TRAIN_ITERS=${25}
    LR_WARMUP_ITERS=${26}
    LR_DECAY_ITERS=$(( ${TRAIN_ITERS} - ${LR_WARMUP_ITERS}))
    PREFIX="finetune-mcore-qwen2-megatron-${MODEL_SIZE}-lr-${LR}-minlr-${MIN_LR}-bs-${BATCH_SIZE}-gbs-${GLOBAL_BATCH_SIZE}-seqlen-${SEQ_LEN}"
    NAME=sft-${NAME}
    sft_options=" \
         --eod-mask-loss \
         --calculate-per-token-loss \
         --train-mode finetune"
else
    TRAIN_ITERS=$(( ${TRAIN_TOKENS} / ${GLOBAL_BATCH_SIZE} / ${SEQ_LEN} ))
    LR_WARMUP_ITERS=$(( ${WARMUP_TOKENS}  / ${GLOBAL_BATCH_SIZE} / ${SEQ_LEN} ))
    LR_DECAY_ITERS=$(( ${TRAIN_TOKENS} /  ${GLOBAL_BATCH_SIZE} / ${SEQ_LEN} ))
    PREFIX="pretrain-mcore-qwen2-megatron-${MODEL_SIZE}-lr-${LR}-minlr-${MIN_LR}-bs-${BATCH_SIZE}-gbs-${GLOBAL_BATCH_SIZE}-seqlen-${SEQ_LEN}"
    sft_options=" \
        --train-mode pretrain"
fi

if [ ${MP_DATASET_TYPE} = "raw" ]; then
    dataset_options=" \
        --train-data-path ${DATASET_PATH} \
        --valid-data-path ${VALID_DATASET_PATH} \
        --dataloader-type cyclic \
        --dataset JSON-SFT"
else 
    dataset_options=" \
        --data-path ${DATASET_PATH} \
        --split 100,0,0 \
        --dataset MMAP"
fi

if [ ${MP_SFT_PACKING} = true ]; then
    packing_options=" \
      --reset-position-ids \
      --no-create-attention-mask-in-dataloader
    "
else
    packing_options=""
fi

##### Prepare logdirs #######
mkdir -p "${OUTPUT_BASEPATH}/tensorboard/"
mkdir -p "${OUTPUT_BASEPATH}/checkpoint/"
mkdir -p "${OUTPUT_BASEPATH}/log/"
# TENSORBOARD_DIR="${OUTPUT_BASEPATH}/tensorboard/${NAME}_${current_time}"
TENSORBOARD_DIR="${OUTPUT_BASEPATH}/tensorboard/${NAME}"
mkdir -p ${TENSORBOARD_DIR}
SAVED_PRETRAIN_CHECKPOINT_PATH="${OUTPUT_BASEPATH}/checkpoint/${NAME}"

if [[ "$RESUME" == true && -d "${SAVED_PRETRAIN_CHECKPOINT_PATH}" && -f "${SAVED_PRETRAIN_CHECKPOINT_PATH}/latest_checkpointed_iteration.txt" ]]; then
    load_option=" \
            --load ${SAVED_PRETRAIN_CHECKPOINT_PATH} "
fi
#            --no-load-rng \
#            --no-load-optim \


if [ "${NODE_RANK}" = "0" ]; then
    mkdir -p ${SAVED_PRETRAIN_CHECKPOINT_PATH}
    find -L ${PRETRAIN_CHECKPOINT_PATH} -maxdepth 1 -type f -name "*.json" -print0 | xargs -0 cp -t ${SAVED_PRETRAIN_CHECKPOINT_PATH}
    find -L ${PRETRAIN_CHECKPOINT_PATH} -maxdepth 1 -type f -name "merges.txt" -print0 | xargs -0 cp -t ${SAVED_PRETRAIN_CHECKPOINT_PATH}
fi


megatron_options="  \
        --save ${SAVED_PRETRAIN_CHECKPOINT_PATH} \
        --lr ${LR} \
        --min-lr ${MIN_LR} \
        --lr-decay-style cosine \
        --weight-decay 0.1 \
        --adam-beta1 0.9 \
        --adam-beta2 0.95 \
        --clip-grad 1.0 \
        --init-method-std 0.008 \
        --attention-dropout 0.0 \
        --hidden-dropout 0.0 \
        --lr-decay-iters ${LR_DECAY_ITERS} \
        --lr-warmup-iters ${LR_WARMUP_ITERS} \
        --train-iters ${TRAIN_ITERS} \
        --micro-batch-size ${BATCH_SIZE} \
        --global-batch-size ${GLOBAL_BATCH_SIZE} \
        --num-layers ${NUM_LAYERS} \
        --hidden-size ${HIDDEN_SIZE} \
        --num-attention-heads ${NUM_ATTENTION_HEADS} \
        --ffn-hidden-size ${INTERMEDIATE_SIZE} \
        --seq-length ${SEQ_LEN} \
        --max-position-embeddings ${MAX_POSITION_EMBEDDINGS} \
        --max-padding-length ${PAD_LEN} \
        --log-interval 1 \
        --log-throughput \
        --eval-interval $((2*TRAIN_ITERS)) \
        --eval-iters 0 \
        --save-interval ${SAVE_INTERVAL} \
        --tensorboard-queue-size 1 \
        --tensorboard-dir ${TENSORBOARD_DIR} \
        --log-timers-to-tensorboard \
        --log-validation-ppl-to-tensorboard \
        --tensor-model-parallel-size ${TP} \
        --pipeline-model-parallel-size ${PP} \
        --context-parallel-size ${CP} \
        --num-workers 32 \
        --extra-vocab-size ${EXTRA_VOCAB_SIZE} \
        --patch-tokenizer-type Qwen2Tokenizer \
        --swiglu \
        --normalization RMSNorm \
        --norm-epsilon ${RMS_NORM_EPS} \
        --use-rotary-position-embeddings \
        --position-embedding-type rope \
        --disable-bias-linear \
        --rotary-base ${ROPE_THETA} \
        --ckpt-format torch \
        --transformer-impl transformer_engine \
        --group-query-attention \
        --num-query-groups ${NUM_KEY_VALUE_HEADS} \
        --add-qkv-bias \
        --cross-entropy-loss-fusion \
        --G-I ${G_I} \
        --G-O ${G_O} \
        --R-I ${R_I} \
        --R-O ${R_O} \
        "

run_cmd="torchrun $DISTRIBUTED_ARGS pretrain.py
 ${megatron_options} ${dataset_options} ${pr_options} ${load_option} ${activation_checkpoint_options} \
 ${do_option} ${sp_option} ${moe_options} ${offload_option} ${sft_options} ${vp_option} ${packing_options} \
 ${uneven_split_option} ${attn_backend_option} ${tie_option} ${concat_options} \
 ${extra_options}"

echo "export SAVED_PRETRAIN_MCORE_PATH=${SAVED_PRETRAIN_CHECKPOINT_PATH}" > /tmp/env_vars.sh

echo ${run_cmd}
eval ${run_cmd}
set +x

