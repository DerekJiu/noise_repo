import argparse
from util import Args
from .segmentation import UNet
import timm

# Model registry for segmentation models
SEGMENTATION_MODEL_REGISTRY = {
    "UNet": UNet,
    # Add more segmentation models here
}

def get_model(model_args):
    model_args_ = model_args

    # Convert Namespace to Args if necessary
    if isinstance(model_args, argparse.Namespace):
        model_args_ = Args(vars(model_args))

    # Validate required arguments
    required_args = ["task", "model"]
    for arg in required_args:
        if arg not in model_args_:
            raise ValueError(f"Missing required argument: {arg}")

    model = None

    # Handle classification task
    if model_args_.get('task') == "classification":
        model_name = model_args_.get('model')
        pretrained = model_args_.get('pretrained', True)
        num_classes = model_args_.get('num_classes', 1)
        img_size = model_args_.get('img_size', None)
        timm_img_size_param = model_args_.get('timm_img_size_param', False)

        if timm_img_size_param:
            model = timm.create_model(
                model_name, pretrained=pretrained, num_classes=num_classes, img_size=img_size
            )
        else:
            model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)

    # Handle segmentation task
    elif model_args_.get('task') == "segmentation":
        model_name = model_args_.get("model")
        if model_name not in SEGMENTATION_MODEL_REGISTRY:
            raise ValueError(f"Unknown segmentation model: {model_name}")
        model = SEGMENTATION_MODEL_REGISTRY[model_name](model_args_)

    # Raise an error if no model was assigned
    if model is None:
        raise ValueError("Failed to initialize a model. Ensure task and model type are correctly specified.")

    return model