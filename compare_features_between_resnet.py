import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA

# Cosine Similarity 계산 (numpy 배열 -> torch 이용)
def cosine_similarity_np(feature1, feature2):
    f1 = torch.tensor(feature1).flatten().float()
    f2 = torch.tensor(feature2).flatten().float()
    return F.cosine_similarity(f1.unsqueeze(0), f2.unsqueeze(0)).item()

# L1 차이 계산 (numpy)
def l1_distance_np(feature1, feature2):
    return np.sum(np.abs(feature1 - feature2))

# PCA를 활용해 (C,H,W) feature map을 2D 이미지로 축소하고, [0,1] 범위로 정규화
def reduce_feature_map(feature_map):
    C, H, W = feature_map.shape
    X = feature_map.reshape(C, -1).T  # shape: (H*W, C)
    pca = PCA(n_components=1, svd_solver='randomized')
    projection = pca.fit_transform(X).squeeze()  # shape: (H*W,)
    projection = projection.reshape(H, W)
    proj_min, proj_max = projection.min(), projection.max()
    if proj_max - proj_min > 0:
        projection_norm = (projection - proj_min) / (proj_max - proj_min)
    else:
        projection_norm = projection
    return projection_norm

# weight 파일 경로에서 basename 추출 (예: "modules/pascal_fold1.ckpt" -> "pascal")
def get_weight_basename(weight_path):
    filename = os.path.basename(weight_path)
    pos = filename.find('_')
    if pos == -1:
        return filename.split('.')[0]
    return filename[:pos]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True,
                        help="Input image file (e.g., images/cat.jpeg)")
    parser.add_argument("--sizes", type=int, nargs='+', default=[480,384,320,256],
                        help="List of image sizes (e.g., 480 384 320 256)")
    parser.add_argument("--weights", type=str, nargs='+', 
                        default=["modules/pascal_fold1.ckpt", "modules/coco_fold1.ckpt", "modules/fss_rn101.ckpt"],
                        help="List of weight files (e.g., modules/pascal_fold1.ckpt modules/coco_fold1.ckpt modules/fss_rn101.ckpt)")
    args = parser.parse_args()

    image_name = os.path.basename(args.image).split('.')[0]
    sizes = args.sizes
    weight_list = args.weights

    # 각 size, 각 dataset(weight)의 PT feature map들을 로드
    # 파일명 규칙: outputs/feature_map/resnet_feature_<image>_<size>_<weightBasename>.npy
    features = {}  # features[size][weight_basename] = PT feature map (shape: (C,H,W))
    for size in sizes:
        features[size] = {}
        for weight in weight_list:
            weight_basename = get_weight_basename(weight)
            pt_path = os.path.join("outputs/feature_map", f"resnet_feature_{image_name}_{size}_{weight_basename}.npy")
            if not os.path.exists(pt_path):
                print(f"[WARNING] PT feature map not found: {pt_path}")
                continue
            feat = np.load(pt_path)  # assume shape: (1, C, H, W)
            features[size][weight_basename] = feat[0]

    # Pairwise 비교: 각 size별로 각 dataset 간의 PT feature map 비교 (pascal vs coco, pascal vs fss, coco vs fss)
    pairwise_results = {}
    for size in sizes:
        if size not in features:
            continue
        pairwise_results[size] = {}
        weight_names = list(features[size].keys())
        for i in range(len(weight_names)):
            for j in range(i+1, len(weight_names)):
                w1 = weight_names[i]
                w2 = weight_names[j]
                feat1 = features[size][w1]
                feat2 = features[size][w2]
                cos_sim = cosine_similarity_np(feat1, feat2)
                l1_diff = l1_distance_np(feat1, feat2)
                pairwise_results[size][(w1, w2)] = (cos_sim, l1_diff)

    # 결과 테이블 출력 (보기 좋게 정렬)
    header = f"{'Size':>6} | {'Dataset Pair':>20} | {'Cosine':>8} | {'L1':>8}"
    print("\nPairwise Dataset Comparison (PT Only):")
    print(header)
    print("-" * len(header))
    for size in sorted(pairwise_results.keys()):
        for (w1, w2), metrics in pairwise_results[size].items():
            print(f"{size:6d} | {w1 + ' vs ' + w2:20} | {metrics[0]:8.3f} | {metrics[1]:8.1f}")
