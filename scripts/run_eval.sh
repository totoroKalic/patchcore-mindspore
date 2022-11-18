
if [ $# != 4 ]
then
    echo "=============================================================================================================="
    echo "Please run the script as: "
    echo "bash run_eval.sh DATA_PATH DEVICE_ID PRETRAINED_PATH CATEGORY"
    echo "For example: bash run_eval.sh /path/dataset /path/pretrained_path category 0"
    echo "It is better to use the absolute path."
    echo "=============================================================================================================="
exit 1
fi
set -e

get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}
DATA_PATH=$(get_real_path $1)
CKPT_APTH=$(get_real_path $2)
export DATA_PATH=$DATA_PATH

eval_path=eval_$3
if [ -d $eval_path ];
then
    rm -rf ./$eval_path
fi
mkdir ./$eval_path
cd ./$eval_path
env > env0.log
echo "[INFO] start eval dataset $3."
python ../../eval.py --dataset_path $DATA_PATH --pre_ckpt_path $CKPT_APTH --category $3  --device_id $4 &> eval.log

if [ $? -eq 0 ];then
    echo "[INFO] eval success"
else
    echo "[ERROR] eval failed"
    exit 2
fi
echo "[INFO] finish"
cd ../
