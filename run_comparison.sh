#!/bin/bash

# 이미지 파일 리스트
IMAGES=("images/cat.jpeg" "images/cat2.jpeg" "images/cat3.jpeg")

# 사용할 이미지 크기 (내림차순 정렬)
#SIZES=(480 384 320 128)
SIZES=(384 320 320 256)

WEIGHTS=("modules/pascal_fold1.ckpt" "modules/coco_fold1.ckpt" "modules/fss_rn101.ckpt")
for IMAGE_FILE in "${IMAGES[@]}"; do
    echo "✅ Running Pytorch Model for image: $IMAGE_FILE with sizes: ${SIZES[*]}"
    python3 model_output_zs.py --images "$IMAGE_FILE" --sizes ${SIZES[*]} --weights ${WEIGHTS[*]}

    echo "✅ Running TensorRT Inference for image: $IMAGE_FILE"
    ./cpp_project/model_output/build/trt_feature_extractor "$IMAGE_FILE" ${SIZES[*]} --weights ${WEIGHTS[*]}

    echo "✅ Comparing Features for image: $IMAGE_FILE"
    python3 compare_features.py --image "$IMAGE_FILE" --sizes ${SIZES[*]} --weights ${WEIGHTS[*]}

done

echo "✅ All image comparisons completed!"