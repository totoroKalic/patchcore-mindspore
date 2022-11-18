
if [ $# != 3 ]
then
    echo "=============================================================================================================="
    echo "Please run the script as: "
    echo "bash run_all_mvtec.sh DATA_PATH DEVICE_ID PRETRAINED_PATH CATEGORY"
    echo "For example: bash run_all_mvtec.sh /path/dataset /path/pretrained_path 0"
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

arr=("bottle" "cable" "capsule" "carpet" "grid" "hazelnut" "leather" "metal_nut" "pill" "screw" "tile" "toothbrush" "transistor" "wood" "zipper")

for value in "${arr[@]}"
do
  bash run_standalone_train.sh  $DATA_PATH  $CKPT_APTH $value $3
  bash run_eval.sh  $DATA_PATH  $CKPT_APTH $value $3
done

img_auc=$(grep "auc" eval_*/eval.log | awk -F "img_auc:" '{print $2}' | awk -F "," '{print $1}' | awk '{sum+=$1}END{print sum/NR}' | awk '{printf("%.3f", $1)}')
echo "[INFO] average img_auc = $img_auc"

pixel_auc=$(grep "auc" eval_*/eval.log | awk -F "img_auc:" '{print $2}' | awk -F "pixel_auc:" '{print $2}' | awk '{sum+=$1}END{print sum/NR}' | awk '{printf("%.3f", $1)}')
echo "[INFO] average pixel_auc = $pixel_auc"
