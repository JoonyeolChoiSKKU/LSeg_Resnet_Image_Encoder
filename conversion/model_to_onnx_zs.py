import argparse
import torch
import torch.onnx
from modules.lseg_module_zs import LSegModuleZS
import os
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_size", type=int, default=480, help="Input image size")
    parser.add_argument("--weights", type=str, default="modules/pascal_fold0.ckpt", help="Path to checkpoint")
    parser.add_argument("--datapath", type=str, default="data/", help="Path to data (for label file loading)")
    parser.add_argument("--dataset", type=str, default="ade20k", help="Dataset name")
    parser.add_argument("--backbone", type=str, default="clip_resnet101", help="Backbone network (e.g., clip_resnet101)")
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
    parser.add_argument("--activation", type=str, default="lrelu", choices=["relu", "lrelu", "tanh"], help="Activation function")
    args = parser.parse_args()

    # LSegModuleZS로부터 모델 불러오기 (이미지 인코더 부분만 사용)
    module = LSegModuleZS.load_from_checkpoint(
        checkpoint_path=args.weights,
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
    )
    
    # ONNX export를 위해 모델의 이미지 인코더(net)만 eval 모드로 설정 (CPU 사용)
    model = module.net.eval()

    dummy_input = torch.randn(1, 3, args.img_size, args.img_size)
    onnx_filename = f"output/models/lseg_img_enc_rn101_" + \
                    f"{os.path.basename(args.weights).split('_')[0]}_"+ \
                    f"{args.img_size}.onnx"
    
    torch.onnx.export(
        model,
        dummy_input,
        onnx_filename,
        input_names=["input"],
        output_names=["output"],
        opset_version=14,
        dynamic_axes=None
    )
    
    print(f"✅ Image Encoder가 ONNX 파일로 저장되었습니다: {onnx_filename}")
