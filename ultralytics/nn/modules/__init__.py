# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
Ultralytics modules.

Example:
    Visualize a module with Netron.
    ```python
    from ultralytics.nn.modules import *
    import torch
    import os

    x = torch.ones(1, 128, 40, 40)
    m = Conv(128, 128)
    f = f'{m._get_name()}.onnx'
    torch.onnx.export(m, x, f)
    os.system(f'onnxsim {f} {f} && open {f}')
    ```
"""

from .block import (
    C1,
    C2,
    C3,
    C3TR,
    DFL,
    SPP,
    SPPELAN,
    SPPF,
    ADown,
    BNContrastiveHead,
    Bottleneck,
    BottleneckCSP,
    C2f,
    C2fAttn,
    C3Ghost,
    C3x,
    CBFuse,
    CBLinear,
    ContrastiveHead,
    GhostBottleneck,
    HGBlock,
    HGStem,
    ImagePoolingAttn,
    Proto,
    RepC3,
    RepNCSPELAN4,
    ResNetLayer,
    Silence,BiLevelRoutingAttention,C2f_EffectiveSE, C2f_GlobalContext, C2f_GatherExcite, C2f_MHSA,ScConv,
    DCNV2,C2f_SimAM,CARAFE,C2f_CA,C2f_ECA,C2f_SE,C2f_CBAM,DAttentionBaseline,BiFPN_Add2, BiFPN_Add3,C2f_CoT, C2f_Double, C2f_SK,CBRM, Shuffle_Block,
MV2Block, MobileViTBlock,GhostV2,stem, MBConvBlock,PatchMerging, PatchEmbed, SwinStage,ResNetLayer,Conv_BN_HSwish, MobileNetV3_InvertedResidual,

)
from .conv import (
    CBAM,
    ChannelAttention,
    Concat,
    Conv,
    Conv2,
    ConvTranspose,
    DWConv,
    DWConvTranspose2d,
    Focus,
    GhostConv,
    LightConv,
    RepConv,
    SpatialAttention,
    BiFPN_Concat2,
    BiFPN_Concat3,CBAM,BiFormerBlock,C2f_BiLevelRoutingAttention,C3_BiLevelRoutingAttention,SPDConv,
)
from .head import OBB, Classify, Detect, Pose, RTDETRDecoder, Segment, WorldDetect
from .transformer import (
    AIFI,
    MLP,
    DeformableTransformerDecoder,
    DeformableTransformerDecoderLayer,
    LayerNorm2d,
    MLPBlock,
    MSDeformAttn,
    TransformerBlock,
    TransformerEncoderLayer,
    TransformerLayer,
)

__all__ = (
    "Conv",
    "Conv2",
    "LightConv",
    "RepConv",
    "DWConv",
    "DWConvTranspose2d",
    "ConvTranspose",
    "Focus",
    "GhostConv",
    "ChannelAttention",
    "SpatialAttention",
    "CBAM",
    "Concat",
    "TransformerLayer",
    "TransformerBlock",
    "MLPBlock",
    "LayerNorm2d",
    "DFL",
    "HGBlock",
    "HGStem",
    "SPP",
    "SPPF",
    "C1",
    "C2",
    "C3",
    "C2f",
    "C2fAttn",
    "C3x",
    "C3TR",
    "C3Ghost",
    "GhostBottleneck",
    "Bottleneck",
    "BottleneckCSP",
    "Proto",
    "Detect",
    "Segment",
    "Pose",
    "Classify",
    "TransformerEncoderLayer",
    "RepC3",
    "RTDETRDecoder",
    "AIFI",
    "DeformableTransformerDecoder",
    "DeformableTransformerDecoderLayer",
    "MSDeformAttn",
    "MLP",
    "ResNetLayer",
    "OBB",
    "WorldDetect",
    "ImagePoolingAttn",
    "ContrastiveHead",
    "BNContrastiveHead",
    "RepNCSPELAN4",
    "ADown",
    "SPPELAN",
    "CBFuse",
    "CBLinear",
    "Silence",
    "BiLevelRoutingAttention",
    "ScConv",
    "DCNV2",
    "BiFPN_Concat2",
    "BiFPN_Concat3","BiFormerBlock","C2f_BiLevelRoutingAttention","C3_BiLevelRoutingAttention",
    "CARAFE","SPDConv","C2f_CA","C2f_SE","C2f_ECA","C2f_CBAM","DAttentionBaseline","BiFPN_Add2","BiFPN_Add3","C2f_CoT","C2f_Double","C2f_SK","CBRM", "Shuffle_Block",
    "MV2Block","MobileViTBlock","GhostV2",'stem', 'MBConvBlock','PatchMerging', 'PatchEmbed', 'SwinStage','ResNetLayer','Conv_BN_HSwish', 'MobileNetV3_InvertedResidual',

)
