MCORE_PATH=${1}
model_size=${2}
MODELING_PATH=${3} 
DENSE_ARCH_PATH=${4}
BASE_PATH=${5}

MODEL_NAME=$(basename "${MCORE_PATH}")
MODEL_NAME="${MODEL_NAME%%-DataMix*}"  

HF_PATH=${BASE_PATH}/mcore2hf/${MODEL_NAME}

echo "export HF_PATH=${HF_PATH}" > /tmp/env_vars.sh


mkdir -p ${HF_PATH}
rm ${HF_PATH}/convert_done
rm ${HF_PATH}/model*

if [[ "${RANK}" -gt 0 ]]; then
    echo "Rank-${RANK} waits to convert HF from mcore"
    exit 0
fi

########################
# BUILD HF_CKPT_PATH
########################
HF_ARCH_PATH=${BASE_PATH}/output/hf_arch/${MODEL_NAME}
echo -e "\n******************"
echo "Building hf architecture to $HF_ARCH_PATH"
echo -e "******************\n"
cd toolkits/model_checkpoints_convertor/FineRMoE/ckpt_convert_utils
bash build_hf_ckpt.sh ${HF_ARCH_PATH} ${MODELING_PATH} ${DENSE_ARCH_PATH}

########################
# MCORE -> HF
########################

echo -e "\n******************"
echo "Converting from mcore to hf: ${MCORE_PATH} -> ${HF_PATH}"
echo -e "******************\n"
MG2HF=true

cd toolkits/model_checkpoints_convertor/FineRMoE/ckpt_convert_utils/
bash mcore2hf_finermoe.sh \
${MCORE_PATH} \
${HF_PATH} \
${HF_ARCH_PATH} \
${MG2HF} \
${model_size}

echo "" > ${HF_PATH}/convert_done