"""Feature specifications for the additional Science Advances backbone sweep.

The original backbone implementations (IDs 0-6) remain in their existing callers.
"""

FEATURE_DIMS = {5: 768, 7: 1280, 8: 768, 9: 2048, 10: 768,
                11: 768, 12: 512, 13: 512}


def feature_dim(bbv):
    """Keep the original 2048-dimensional default for the existing backbones."""
    return FEATURE_DIMS.get(bbv, 2048)


def load_extended_backbone(bbv, pretrained=True):
    """Return model, preprocessing, feature layer and CLS-token flag for IDs 7-13.

    These weights, transforms, and readouts follow the Science Advances analyses.
    CORnet is imported only when its backbone is selected.
    """
    import torch
    from torchvision import models, transforms

    specs = {
        7: ('efficientnet_b1', 'EfficientNet_B1_Weights', 'IMAGENET1K_V1'),
        8: ('vit_b_32', 'ViT_B_32_Weights', 'IMAGENET1K_V1'),
        9: ('resnet101', 'ResNet101_Weights', 'IMAGENET1K_V2'),
        10: ('convnext_tiny', 'ConvNeXt_Tiny_Weights', 'IMAGENET1K_V1'),
        11: ('swin_t', 'Swin_T_Weights', 'IMAGENET1K_V1'),
        12: ('maxvit_t', 'MaxVit_T_Weights', 'IMAGENET1K_V1'),
    }
    if bbv == 13:
        import cornet
        model = cornet.cornet_s(pretrained=pretrained, map_location=torch.device('cpu'))
        preprocess = transforms.Compose([
            transforms.Resize(224, antialias=True),
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        layer = model.module.decoder.avgpool
    elif bbv in specs:
        constructor, enum_name, version = specs[bbv]
        weights = getattr(getattr(models, enum_name), version)
        model = getattr(models, constructor)(weights=weights if pretrained else None)
        preprocess = weights.transforms()
        layer = model.encoder if bbv == 8 else model.classifier[0] if bbv == 12 else model.avgpool
    else:
        raise ValueError(f'Expected an additional paper backbone ID (7-13), got {bbv}')
    return model, preprocess, layer, bbv == 8


def forward_features(model, layer, images, cls_token=False):
    """Capture the specified readout without leaving a persistent forward hook."""
    captured = {}

    def capture(_module, _inputs, output):
        captured['features'] = output

    handle = layer.register_forward_hook(capture)
    try:
        model(images)
    finally:
        handle.remove()
    features = captured['features']
    return features[:, 0, :] if cls_token else features.flatten(1)
