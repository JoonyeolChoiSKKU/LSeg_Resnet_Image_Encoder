import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import os
import argparse
from sklearn.decomposition import PCA

# Cosine Similarity 계산
def cosine_similarity(feature1, feature2):
    f1 = torch.tensor(feature1).flatten().float()
    f2 = torch.tensor(feature2).flatten().float()
    return F.cosine_similarity(f1.unsqueeze(0), f2.unsqueeze(0)).item()

# 기본적으로 L1 차이를 계산하는 함수
def l1_distance_np(feature1, feature2):
    return np.sum(np.abs(feature1 - feature2))

# L2 Distance 계산
def l2_distance(feature1, feature2):
    return np.linalg.norm(feature1 - feature2)


def reduce_feature_map(feature_map):
    """
    feature_map: numpy array of shape (C, H, W)
    모든 채널의 정보를 이용해 각 픽셀의 C차원 벡터를 1차원 값으로 투영하고,
    결과를 HxW 이미지로 반환합니다.
    scikit-learn의 PCA를 활용하여 최적화된 속도로 차원 축소를 수행합니다.
    """
    C, H, W = feature_map.shape
    # 각 픽셀의 feature를 행(row)로 두기 위해 (H*W, C) 형태로 변환합니다.
    X = feature_map.reshape(C, -1).T  # shape: (H*W, C)
    pca = PCA(n_components=1, svd_solver='randomized')
    projection = pca.fit_transform(X).squeeze()  # shape: (H*W,)
    projection = projection.reshape(H, W)
    # [0,1] 범위로 정규화
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
    # 입력 파라미터 설정
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True,
                        help="Input image file (e.g., images/cat.jpeg)")
    parser.add_argument("--sizes", type=int, nargs='+', default=[480,384,320,256],
                        help="List of image sizes (e.g., 480 384 320 256)")
    parser.add_argument("--weights", type=str, nargs='+', 
                        default=["modules/pascal_fold1.ckpt", "modules/coco_fold1.ckpt", "modules/fss_rn101.ckpt"],
                        help="List of weight files (e.g., modules/pascal_fold1.ckpt modules/coco_fold1.ckpt modules/fss_rn101.ckpt)")
    args = parser.parse_args()
    # 비교할 파일 리스트
    sizes = args.sizes
    image_name = os.path.basename(args.image).split('.')[0]
    weight_list = args.weights

    # 총 컬럼 수: 1 (ViT) + 2 per weight (PyTorch, TRT)
    num_cols = 1 + 2 * len(weight_list)
    num_rows = len(sizes)

    # 준비: 파일 경로 규칙
    # ViT feature: outputs/feature_map/pytorch_feature_<image>_<size>.npy
    # Resnet PyTorch: outputs/feature_map/pytorch_feature_<image>_<size>_<weightBasename>.npy
    # Resnet TRT: outputs/feature_map/trt_feature_<weightBasename>_<size>_<image>.npy

    # figure 생성
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols*3, num_rows*3))
    if num_rows == 1:
        axes = axes[np.newaxis, :]
    if num_cols == 1:
        axes = axes[:, np.newaxis]

    # 결과 기록 리스트 (비교 수치)
    comparison_results = {}

    for row_idx, size in enumerate(sizes):
        # Load ViT feature map
        vit_path = os.path.join("outputs/feature_map", f"pytorch_feature_{image_name}_{size}.npy")
        if not os.path.exists(vit_path):
            print(f"[WARNING] ViT feature map not found: {vit_path}")
            continue
        vit_feature = np.load(vit_path)
        # assume shape: (1, C, H, W) -> reduce to (C, H, W)
        vit_feature_map = vit_feature[0]
        vit_reduced = reduce_feature_map(vit_feature_map)
        # Plot ViT in column 0
        axes[row_idx, 0].imshow(vit_reduced, cmap="viridis")
        axes[row_idx, 0].set_title(f"ViT_{size}")
        axes[row_idx, 0].axis("off")
        
        # For each weight, load Resnet feature maps (PyTorch and TRT)
        for w_idx, weight in enumerate(weight_list):
            weight_basename = get_weight_basename(weight)
            # Resnet PyTorch feature map
            resnet_pt_path = os.path.join("outputs/feature_map", f"resnet_feature_{image_name}_{size}_{weight_basename}.npy")
            # Resnet TRT feature map
            resnet_trt_path = os.path.join("outputs/feature_map", f"trt_feature_{weight_basename}_{size}_{image_name}.npy")
            
            if not os.path.exists(resnet_pt_path):
                print(f"[WARNING] Resnet PyTorch feature map not found: {resnet_pt_path}")
                continue
            if not os.path.exists(resnet_trt_path):
                print(f"[WARNING] Resnet TRT feature map not found: {resnet_trt_path}")
                continue
            
            resnet_pt = np.load(resnet_pt_path)
            resnet_trt = np.load(resnet_trt_path)
            resnet_pt_map = resnet_pt[0]
            resnet_trt_map = resnet_trt[0]
            resnet_pt_reduced = reduce_feature_map(resnet_pt_map)
            resnet_trt_reduced = reduce_feature_map(resnet_trt_map)
            
            # Compute metrics relative to ViT
            cos_pt = cosine_similarity(vit_feature_map, resnet_pt_map)
            l1_pt = l1_distance_np(vit_feature_map, resnet_pt_map)
            cos_trt = cosine_similarity(vit_feature_map, resnet_trt_map)
            l1_trt = l1_distance_np(vit_feature_map, resnet_trt_map)
            # Also compute difference between resnet_pt and resnet_trt
            cos_diff = cosine_similarity(resnet_pt_map, resnet_trt_map)
            l1_diff = l1_distance_np(resnet_pt_map, resnet_trt_map)
            
            # Column indices: for weight w_idx, pytorch in col = 1 + 2*w_idx, trt in col = 1 + 2*w_idx + 1
            col_pt = 1 + 2*w_idx
            col_trt = col_pt + 1
            
            # Plot Resnet PyTorch
            axes[row_idx, col_pt].imshow(resnet_pt_reduced, cmap="viridis")
            axes[row_idx, col_pt].set_title(f"Resnet_{weight_basename}_pt_{size}\nCos:{cos_pt:.3f} L1:{l1_pt:.1f}")
            axes[row_idx, col_pt].axis("off")
            
            # Plot Resnet TRT
            axes[row_idx, col_trt].imshow(resnet_trt_reduced, cmap="viridis")
            axes[row_idx, col_trt].set_title(f"Resnet_{weight_basename}_trt_{size}\nCos:{cos_trt:.3f} L1:{l1_trt:.1f}")
            axes[row_idx, col_trt].axis("off")
            
            # 기록 저장 (선택 사항)
            comparison_results[(size, weight_basename)] = {
                "ViT vs PT": (cos_pt, l1_pt),
                "ViT vs TRT": (cos_trt, l1_trt),
                "PT vs TRT": (cos_diff, l1_diff)
            }

    # 최종 비교 결과를 보기 좋게 테이블 형식으로 출력
    header = f"{'Size':>6} | {'Weight':>10} | {'ViT vs PT Cos':>15} | {'ViT vs PT L1':>15} | {'ViT vs TRT Cos':>15} | {'ViT vs TRT L1':>15} | {'PT vs TRT Cos':>15} | {'PT vs TRT L1':>15}"
    print("\nComparison Results:")
    print(header)
    print("-" * len(header))
    for key, vals in sorted(comparison_results.items()):
        size, weight = key
        print(f"{size:6d} | {weight:10} | {vals['ViT vs PT'][0]:15.3f} | {vals['ViT vs PT'][1]:15.1f} | {vals['ViT vs TRT'][0]:15.3f} | {vals['ViT vs TRT'][1]:15.1f} | {vals['PT vs TRT'][0]:15.3f} | {vals['PT vs TRT'][1]:15.1f}")

    plt.tight_layout()
    plt.show()
    
    