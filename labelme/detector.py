from torch import device, load, Tensor
from torch import nn
from torchvision.models.detection import MaskRCNN
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork, LastLevelMaxPool
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.models import convnext_base, ConvNeXt_Base_Weights, convnext_tiny, ConvNeXt_Tiny_Weights, convnext_small, ConvNeXt_Small_Weights, convnext_large, ConvNeXt_Large_Weights
import numpy as np
from PIL import Image


class BackboneWithFPN(nn.Module):
    """
    Adds a FPN on top of a model.
    Internally, it uses torchvision.models.feature_extraction.create_feature_extractor 
    to extract a submodel that returns the feature maps specified in given backbone 
    feature extractor model.
    Parameters
    ----------
    backbone (nn.Module): Feature extractor ConvNeXt pretrained model.        
    in_channels_list (List[int]): Number of channels for each feature map
        that is returned, in the order they are present in the OrderedDict
    out_channels (int): number of channels in the FPN.
    norm_layer (callable, optional): Default None.
        Module specifying the normalization layer to use. 
    extra_blocks (callable, optional): Default None.
        Extra optional FPN blocks.
    Attributes
    ----------
    out_channels : int
        The number of channels in the FPN.
    """

    def __init__(
        self,
        backbone,
        in_channels_list,
        out_channels,
        extra_blocks = None,
        norm_layer = None,
    ):
        super().__init__()

        if extra_blocks is None:
            extra_blocks = LastLevelMaxPool()

        self.body = backbone
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=out_channels,
            extra_blocks=extra_blocks,
            norm_layer=norm_layer,
        )
        self.out_channels = out_channels

    def forward(self, x):
        x = self.body(x)
        x = self.fpn(x)
        return x


def convnext_fpn_backbone(
    backbone_name,
    trainable_layers,
    extra_blocks = None,
    norm_layer = None,
    feature_dict = {'1': '0', '3': '1', '5':'2', '7':'3'},
    out_channels = 256
):
    """
    Returns an FPN-extended backbone network using a feature extractor 
    based on models developed in the article 'A ConvNet for the 2020s'.
    For detailed information about the feature extractor ConvNeXt, read the article.
    https://arxiv.org/abs/2201.03545
    Parameters
    ----------
    backbone_name : str
        ConvNeXt architecture. Possible values are 'convnext_tiny', 'convnext_small', 
        'convnext_base' or 'convnext_large'.
    trainable_layers : int
        Number of trainable (not frozen) layers starting from final block.
        Valid values are between 0 and 8, with 8 meaning all backbone layers 
        are trainable.
    extra_blocks (ExtraFPNBlock or None): default a ``LastLevelMaxPool`` is used.
        If provided, extra operations will be performed. It is expected to take 
        the fpn features, the original features and the names of the original 
        features as input, and returns a new list of feature maps and their 
        corresponding names.
    norm_layer (callable, optional): Default None.
        Module specifying the normalization layer to use. It is recommended to use 
        the default value. For details visit: 
        (https://github.com/facebookresearch/maskrcnn-benchmark/issues/267) 
    feature_dict : dictionary
        Contains the names of the 'nn.Sequential' object used in the ConvNeXt 
        model configuration if you need more detailed information, 
        https://github.com/facebookresearch/ConvNeXt. 
    out_channels (int): defaults to 256.
        Number of channels in the FPN.
    Returns
    -------
    BackboneWithFPN : torch.nn.Module
        Returns a specified ConvNeXt backbone with FPN on top. 
        Freezes the specified number of layers in the backbone.
    """
    input_channels_dict = {
        'convnext_tiny': [96, 192, 384, 768],
        'convnext_small': [96, 192, 384, 768],
        'convnext_base': [128, 256, 512, 1024],
        'convnext_large': [192, 384, 768, 1536],
        'convnext_xlarge': [256, 512, 1024, 2048],
    }
    if backbone_name == "convnext_tiny":
        backbone = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT).features
        backbone = create_feature_extractor(backbone, feature_dict)
    elif backbone_name == "convnext_small":
        backbone = convnext_small(weights=ConvNeXt_Small_Weights.DEFAULT).features
        backbone = create_feature_extractor(backbone, feature_dict)
    elif backbone_name == "convnext_base":
        backbone = convnext_base(weights=ConvNeXt_Base_Weights.DEFAULT).features
        backbone = create_feature_extractor(backbone, feature_dict)
    elif backbone_name == "convnext_large":
        backbone = convnext_large(weights=ConvNeXt_Large_Weights.DEFAULT).features
        backbone = create_feature_extractor(backbone, feature_dict)
    else:
        raise ValueError(f"Backbone names should be in the {list(input_channels_dict.keys())}, got {backbone_name}")

    in_channels_list = input_channels_dict[backbone_name]

    # select layers that wont be frozen
    if trainable_layers < 0 or trainable_layers > 8:
        raise ValueError(f"Trainable layers should be in the range [0,8], got {trainable_layers}")
    layers_to_train = ["7", "6", "5", "4", "3", "2", "1"][:trainable_layers]
    if trainable_layers == 8:
        layers_to_train.append("bn1")
    for name, parameter in backbone.named_parameters():
        if all([not name.startswith(layer) for layer in layers_to_train]):
            parameter.requires_grad_(False)

    if extra_blocks is None:
        extra_blocks = LastLevelMaxPool()

    return BackboneWithFPN(
        backbone, in_channels_list, out_channels, 
        extra_blocks=extra_blocks, norm_layer=norm_layer
    )


def get_model(model_path, backbone="convnext_base", 
              layers=5, num_classes=2, max_size=512, min_size=512):
    """
    Get pretrained MaskRCNN model with ConvNeXt backbone.
    Parameters
    ----------
    model_path : (str)
        File path where the pre-trained model is located.
    backbone (str, optional): Defaults to "convnext_base".
        Whichever backbone network is used in the training is entered.
    layers (int, optional): Defaults to 5.
        Number of trainable (not frozen) layers starting from final block.
        Valid values are between 0 and 8, with 8 meaning all backbone layers 
        are trainable. Just for backbone initialization.
    num_classes (int, optional): Defaults to 2.
        Number of classes for prediction. 
        Just for MaskRCNN model initialization.
    max_size (int, optional): Defaults to 512.
        Max image size given input for model.
        Just for MaskRCNN model initialization.
    min_size (int, optional): Defaults to 512.
        Min image size given input for model.
        Just for MaskRCNN model initialization.
    Returns
    -------
    model: torchvision.model
        Pre-trained MaskRCNN model.
    """
    backbone = convnext_fpn_backbone(
        backbone,
        layers
    )

    model = MaskRCNN(
        backbone, 
        num_classes=num_classes, 
        max_size=max_size,
        min_size=min_size,
    )
    model.load_state_dict(
        load(model_path, map_location=device('cpu'))["model_state_dict"]
    )
    model.eval()
    return model


def get_prediction(model, img_path, confidence, CLASS_NAMES, mask_th=0.5):
    """
    It reads the image from the given file path and 
    returns the output predicted by the model.
    Parameters
    ----------
    model : pytorch model
        The model to generate the predictions.
    img_path : str
        The file path with the image to be predicted.
    confidence : float
        Enter a value between 0 and 1 to access reliable results. 
        A value close to 1 is the result that the model predicted 
        with higher confidence.
    CLASS_NAMES : list
        Used to return class names based on a list of label names.
    mask_th (float, optional): Defaults to 0.5.
        Enter a value between 0 and 1 to threshold the probability 
        map generated by the model.
    Returns
    ----------
    masks : np.uint8
        Thresholded masks according to the given threshold value.
    pred_boxes : list
        Bounding boxes predicted by the model.
    pred_class : list
        Name of predicted classes.
    """
    img = Image.open(img_path).convert('RGB')
    img = Tensor(np.array(img, dtype=np.uint8).transpose((2, 0, 1)))
    img = img.to("cpu")
    pred = model([img])
    pred_score = list(pred[0]['scores'].detach().cpu().numpy())
    while True:
        try:
            indices = [pred_score.index(x) for x in pred_score if x>confidence][-1]
            break
        except:
            confidence -= 0.1
    masks = (pred[0]['masks']>mask_th).squeeze().detach().cpu().numpy()
    pred_boxes = [[(i[1], i[0]), (i[3], i[2])] for i in list(pred[0]['boxes'].detach().cpu().numpy())]
    pred_class = [CLASS_NAMES[i] for i in list(pred[0]['labels'].cpu().numpy())]
    masks = masks[:indices+1]
    pred_boxes = pred_boxes[:indices+1]
    pred_class = pred_class[:indices+1]
    return pred_boxes, masks, pred_class