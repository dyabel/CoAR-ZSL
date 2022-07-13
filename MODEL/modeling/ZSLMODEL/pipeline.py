from .ZSLNet import build_ZSLNet
from .ZSLNet_VIT import build_ZSLNet_VIT

_ZSL_META_ARCHITECTURES = {
    "ZSLModel": build_ZSLNet,
    "ZSLModel_VIT": build_ZSLNet_VIT,
}

def build_zsl_pipeline(cfg):
    meta_arch = _ZSL_META_ARCHITECTURES[cfg.MODEL.META_ARCHITECTURE]
    return meta_arch(cfg)