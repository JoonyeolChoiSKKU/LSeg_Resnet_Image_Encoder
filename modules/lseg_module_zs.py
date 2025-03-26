import os
import torch
from .models.lseg_net_zs import LSegNetZS, LSegRNNetZS
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
# upsampling 파라미터 (필요 시 대체 가능)
up_kwargs = {'mode': 'bilinear', 'align_corners': True}

class LSegModuleZS:
    def __init__(self, crop_size=384, **kwargs):
        self.crop_size = crop_size
        # 필요한 경우, 고정된 라벨 리스트 대신 파일에서 라벨 로드
        labels = self.get_labels('ade20k')
        
        # backbone 인자에 따라 ResNet 기반과 그 외의 네트워크 선택
        if kwargs["backbone"] in ["clip_resnet101"]:
            self.net = LSegRNNetZS(
                label_list=labels,
                backbone=kwargs["backbone"],
                features=kwargs["num_features"],
                aux=False,  # 이미지 인코더 추출에는 aux 사용하지 않음
                use_pretrained=kwargs["use_pretrained"],
                arch_option=kwargs["arch_option"],
                block_depth=kwargs["block_depth"],
                activation=kwargs["activation"],
            )
        else:
            self.net = LSegNetZS(
                label_list=labels,
                backbone=kwargs["backbone"],
                features=kwargs["num_features"],
                aux=False,
                use_pretrained=kwargs["use_pretrained"],
                arch_option=kwargs["arch_option"],
                block_depth=kwargs["block_depth"],
                activation=kwargs["activation"],
            )
        # 만약 transformer 백본(ViT 등)을 사용한다면, patch embedding의 이미지 사이즈 설정
        # if hasattr(self.net.pretrained.model, 'patch_embed'):
        #     self.net.pretrained.model.patch_embed.img_size = (self.crop_size, self.crop_size)
        self._up_kwargs = up_kwargs

    def get_labels(self, dataset):
        labels = []
        path = 'label_files/fewshot_{}.txt'.format(dataset)
        if os.path.exists(path):
            with open(path, 'r') as f:
                lines = f.readlines()
            for line in lines:
                labels.append(line.strip())
        return labels

    def forward(self, x):
        """
        입력 x에 대해 이미지 인코더를 통해 feature map을 추출합니다.
        (LSegNetZS 또는 LSegRNNetZS의 forward()가 이미지 인코딩 부분만 수행하도록 구현되어 있다고 가정)
        """
        return self.net.forward(x)

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path, map_location, **kwargs):
        # 안전하게 checkpoint 내 ModelCheckpoint 글로벌을 허용
        torch.serialization.add_safe_globals([ModelCheckpoint])
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        if "model" in checkpoint:
            state = checkpoint["model"]
        elif "state_dict" in checkpoint:
            state = checkpoint["state_dict"]
        else:
            state = checkpoint

        instance = cls(**kwargs)
        new_state = {}
        for key, value in state.items():
            new_key = key
            if key.startswith("net."):
                new_key = key[len("net."):]
            new_state[new_key] = value
        instance.net.load_state_dict(new_state, strict=False)
        return instance

class LSegModuleWrapper:
    def __init__(self, model):
        self.net = model
