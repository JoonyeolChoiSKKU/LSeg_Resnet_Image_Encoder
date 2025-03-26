import torch
import torch.nn as nn
from .lseg_blocks_zs import _make_encoder, Interpolate, FeatureFusionBlock_custom

class BaseModel(nn.Module):
    def load(self, path):
        """Load model from file."""
        parameters = torch.load(path, map_location=torch.device("cpu"))
        if "optimizer" in parameters:
            parameters = parameters["model"]
        self.load_state_dict(parameters)

class LSegNetZS(BaseModel):
    """
    LSegNetZS: Image Encoder 기능만 추출하기 위한 네트워크.
    (ONNX 내보내기 시 forward() 함수만 사용합니다.)
    """
    def __init__(self, label_list, path=None, crop_size=480, **kwargs):
        features = kwargs.get("features", 256)
        kwargs["use_bn"] = True
        self.crop_size = crop_size
        self.label_list = label_list  # 실제 라벨 리스트 (이미지 인코딩에는 사용되지 않음)
        head = nn.Sequential(
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True)
        )
        super(LSegNetZS, self).__init__()
        
        # 백본과 scratch 모듈 생성 (_make_encoder는 backbone 종류에 따라 적절한 구성요소를 반환)
        self.clip_pretrained, self.pretrained, self.scratch = _make_encoder(
            kwargs["backbone"],
            features,
            kwargs["use_pretrained"],
            groups=1,
            expand=False,
            exportable=False,
            hooks=None,
            use_readout=kwargs.get("readout", "ignore"),
        )
        
        # Fusion Block 구성 (scratch 모듈 내에서 RN 변환 후 특징 결합)
        self.scratch.refinenet1 = FeatureFusionBlock_custom(
            features, activation=nn.ReLU(False), deconv=False, bn=True, expand=False, align_corners=True
        )
        self.scratch.refinenet2 = FeatureFusionBlock_custom(
            features, activation=nn.ReLU(False), deconv=False, bn=True, expand=False, align_corners=True
        )
        self.scratch.refinenet3 = FeatureFusionBlock_custom(
            features, activation=nn.ReLU(False), deconv=False, bn=True, expand=False, align_corners=True
        )
        self.scratch.refinenet4 = FeatureFusionBlock_custom(
            features, activation=nn.ReLU(False), deconv=False, bn=True, expand=False, align_corners=True
        )
        
        # 백본에 따른 출력 채널 설정 (ResNet 등은 기본적으로 512로 설정)
        if kwargs["backbone"] in ["clipRN50x16_vitl16_384", "clipRN50x16_vitb32_384"]:
            self.out_c = 768
        elif kwargs["backbone"] in ["clipRN50x4_vitl16_384", "clipRN50x4_vitb32_384"]:
            self.out_c = 640
        else:
            self.out_c = 512
        
        # Projection Layer: scratch 모듈의 최종 채널 수를 조정
        self.scratch.head1 = nn.Conv2d(features, self.out_c, kernel_size=1)
        self.scratch.output_conv = head
        
        if path is not None:
            self.load(path)

    def forward(self, x):
        """
        입력 x에 대해 백본(viT 기반 또는 ResNet 기반)에 따라 이미지 특징(feature map)을 추출합니다.
        """
        # # ViT 기반 백본인 경우 forward_vit()를 사용하고, 그렇지 않으면 ResNet 백본으로 간주
        # if hasattr(self.pretrained, "model"):
        #     layer_1, layer_2, layer_3, layer_4 = forward_vit(self.pretrained, x)
        # else:
        layer_1 = self.pretrained.layer1(x)
        layer_2 = self.pretrained.layer2(layer_1)
        layer_3 = self.pretrained.layer3(layer_2)
        layer_4 = self.pretrained.layer4(layer_3)
        
        # scratch 모듈의 RN 변환 적용
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)
        
        # Fusion Block을 통한 특징 결합 (깊은 레벨부터 fusion)
        path_4 = self.scratch.refinenet4(layer_4_rn)
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn)
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn)
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)
        
        # Projection Layer 적용 및 최종 특징 맵 생성
        image_features = self.scratch.head1(path_1)
        imshape = image_features.shape
        image_features = image_features.permute(0, 2, 3, 1).reshape(-1, self.out_c)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        image_features = image_features.view(imshape[0], imshape[2], imshape[3], self.out_c).permute(0, 3, 1, 2)
        image_features = self.scratch.output_conv(image_features)
        return image_features


class LSegRNNetZS(nn.Module):
    """
    LSegRNNetZS: ResNet 기반 이미지 인코더 추출을 위한 네트워크.
    (ONNX 내보내기나 TorchScript 변환 시 이미지 인코딩 부분만 활용)
    """
    def __init__(self, label_list, path=None, crop_size=480, **kwargs):
        features = kwargs.get("features", 256)
        kwargs["use_bn"] = True  # 항상 BatchNorm 사용
        self.crop_size = crop_size
        self.label_list = label_list  # 이미지 인코딩에는 사용하지 않으나, 체크포인트 로드 시 필요
        head = nn.Sequential(
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True)
        )
        super(LSegRNNetZS, self).__init__()
        
        # 백본과 scratch 모듈 생성: ResNet 기반의 경우 _make_encoder에서 in_shape가 [256, 512, 1024, 2048]로 설정됩니다.
        self.clip_pretrained, self.pretrained, self.scratch = _make_encoder(
            kwargs["backbone"],
            features,
            kwargs["use_pretrained"],
            groups=1,
            expand=False,
            exportable=False,
            hooks=None,
            use_readout=kwargs.get("readout", "ignore"),
        )
        
        # Fusion Block 구성 (깊은 feature부터 결합)
        self.scratch.refinenet1 = FeatureFusionBlock_custom(
            features, activation=nn.ReLU(False), deconv=False, bn=True, expand=False, align_corners=True
        )
        self.scratch.refinenet2 = FeatureFusionBlock_custom(
            features, activation=nn.ReLU(False), deconv=False, bn=True, expand=False, align_corners=True
        )
        self.scratch.refinenet3 = FeatureFusionBlock_custom(
            features, activation=nn.ReLU(False), deconv=False, bn=True, expand=False, align_corners=True
        )
        self.scratch.refinenet4 = FeatureFusionBlock_custom(
            features, activation=nn.ReLU(False), deconv=False, bn=True, expand=False, align_corners=True
        )
        
        # ResNet 기반에서는 기본 출력 채널을 512로 설정
        self.out_c = 512
        # Projection Layer: scratch 모듈의 최종 채널 수를 조정
        self.scratch.head1 = nn.Conv2d(features, self.out_c, kernel_size=1)
        self.scratch.output_conv = head

        if path is not None:
            self.load(path)

    def load(self, path):
        parameters = torch.load(path, map_location=torch.device("cpu"))
        if "optimizer" in parameters:
            parameters = parameters["model"]
        self.load_state_dict(parameters)

    def forward(self, x):
        """
        ResNet 기반 백본에 대해 이미지 인코딩(특징 추출)만 수행.
        """
        # 순차적인 ResNet 레이어 통과 (pretrained는 ResNet backbone 모듈을 구성하고 있음)
        layer_1 = self.pretrained.layer1(x)
        layer_2 = self.pretrained.layer2(layer_1)
        layer_3 = self.pretrained.layer3(layer_2)
        layer_4 = self.pretrained.layer4(layer_3)

        # Scratch 모듈의 RN 변환 적용
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        # Fusion Block을 통한 feature fusion (깊은 레벨부터 결합)
        path_4 = self.scratch.refinenet4(layer_4_rn)
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn)
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn)
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        # Projection Layer 적용: 최종 feature map 생성
        image_features = self.scratch.head1(path_1)
        imshape = image_features.shape
        image_features = image_features.permute(0, 2, 3, 1).reshape(-1, self.out_c)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        image_features = image_features.view(imshape[0], imshape[2], imshape[3], self.out_c).permute(0, 3, 1, 2)
        image_features = self.scratch.output_conv(image_features)
        return image_features