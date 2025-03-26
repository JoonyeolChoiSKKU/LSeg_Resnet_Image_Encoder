import torch
from modules.lseg_module_zs import LSegModuleZS
import os
import numpy as np
from torchvision import transforms
from PIL import Image
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def load_image(image_path, size):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    return transform(image).unsqueeze(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=str, nargs='+', default=["images/cat.jpeg", "images/cat2.jpeg", "images/cat3.jpeg"],
                        help="List of input image files")
    parser.add_argument("--sizes", type=int, nargs='+', default=[480, 384, 320, 256],
                        help="List of image sizes")
    parser.add_argument("--img_size", type=int, default=480, help="Input image size")
    #parser.add_argument("--weights_dir", type=str, default="modules", help="Directory containing ckpt files")
    parser.add_argument("--weights", type=str, nargs='+', required=True,
                        help="List of ckpt weight files (e.g., modules/coco_fold1.ckpt modules/fss_rn101.ckpt)")
    parser.add_argument("--datapath", type=str, default="data/", help="Path to data (for label file loading)")
    parser.add_argument("--dataset", type=str, default="ade20k", help="Dataset name")
    parser.add_argument("--backbone", type=str, default="clip_resnet101",
                        help="Backbone network (e.g., clip_resnet101)")
    parser.add_argument("--aux", action="store_true", help="Auxiliary branch flag")
    parser.add_argument("--ignore_index", type=int, default=255, help="Ignore index")
    parser.add_argument("--scale_inv", action="store_true", help="Scale invariance flag")
    parser.add_argument("--widehead", action="store_true", help="Use wide output head")
    parser.add_argument("--widehead_hr", action="store_true", help="Use wide output head for high resolution")
    parser.add_argument("--arch_option", type=int, default=0, help="Architecture option")
    parser.add_argument("--use_pretrained", type=str, default="True", help="Whether to use pretrained weights")
    parser.add_argument("--strict", action="store_true", help="Strict loading flag")
    parser.add_argument("--fold", type=int, default=0, help="Fold index")
    parser.add_argument("--nshot", type=int, default=1, help="Number of shots")
    parser.add_argument("--activation", type=str, default="lrelu",
                        choices=["relu", "lrelu", "tanh"], help="Activation function")
    args = parser.parse_args()

    # 입력된 ckpt 파일 목록을 그대로 사용합니다.
    weights_files = args.weights

    os.makedirs("outputs", exist_ok=True)

    for weight_path in weights_files:
        weight_name = os.path.basename(weight_path)
        print(f"[INFO] Testing weight file: {weight_name}")
        
        # 모델 로드 (이미지 인코더 부분만 사용)
        model = LSegModuleZS.load_from_checkpoint(
            checkpoint_path=weight_path,
            data_path=args.datapath,
            dataset=args.dataset,
            backbone=args.backbone,
            aux=args.aux,
            num_features=256,
            aux_weight=0,
            se_loss=False,
            se_weight=0,
            base_lr=0,
            batch_size=1,
            max_epochs=0,
            ignore_index=args.ignore_index,
            dropout=0.0,
            scale_inv=args.scale_inv,
            augment=False,
            no_batchnorm=False,
            widehead=args.widehead,
            widehead_hr=args.widehead_hr,
            map_location="cpu",
            arch_option=args.arch_option,
            use_pretrained=args.use_pretrained,
            strict=args.strict,
            logpath='fewshot/logpath_4T/',
            fold=args.fold,
            block_depth=0,
            nshot=args.nshot,
            finetune_mode=False,
            activation=args.activation,
        ).net.to(device)
        
        model.eval()
        
        for image_path in args.images:
            image_basename = os.path.basename(image_path).split('.')[0]
            print(f"[INFO] Processing image: {image_path}")
            for size in args.sizes:
                print(f"[INFO] Testing size: {size}")
                input_tensor = load_image(image_path, size).to(device)
                
                # 입력 텐서를 numpy 파일로 저장
                # input_np = input_tensor.cpu().numpy()
                # input_filename = f"pytorch_input_{image_basename}_{size}.npy"
                # np.save(os.path.join("inputs", input_filename), input_np)
                # print(f"[INFO] Size {size} input tensor saved -> {os.path.join('inputs', input_filename)}")
                
                # 모델 추론 실행
                with torch.no_grad():
                    output = model(input_tensor)
                output_np = output.cpu().numpy()
                
                # 출력 feature map을 numpy 파일로 저장
                output_filename = f"resnet_feature_{image_basename}_{size}_{weight_name.split('_')[0]}.npy"
                output_path = os.path.join("outputs/feature_map", output_filename)
                np.save(output_path, output_np)
                print(f"[INFO] Size {size} feature map saved -> {output_path}")

    print("[INFO] All weight, image, and size feature maps have been saved!")
