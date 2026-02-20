HF_ARCH_PATH=${1}
MODELING_PATH=${2:-"none"}
DENSE_ARCH_PATH=${3}

mkdir -p ${HF_ARCH_PATH}

cp ${DENSE_ARCH_PATH}/*config.json ${HF_ARCH_PATH}/
cp ${DENSE_ARCH_PATH}/tokenizer* ${HF_ARCH_PATH}/
cp ${DENSE_ARCH_PATH}/merges.txt ${HF_ARCH_PATH}/
cp ${DENSE_ARCH_PATH}/vocab.json ${HF_ARCH_PATH}/
cp ${MODELING_PATH}/config.json ${HF_ARCH_PATH}/
cp ${MODELING_PATH}/modeling*.py ${HF_ARCH_PATH}/

echo "MODELING_PATH=${MODELING_PATH}"

cd /toolkits/model_checkpoints_convertor/FineRMoE/ckpt_convert_utils
python convert_json.py --ckpt_path ${HF_ARCH_PATH} --base_arch_path ${DENSE_ARCH_PATH}