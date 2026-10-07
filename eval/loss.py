import torch
import argparse
from monai.losses import DiceLoss
from segmentation_models_pytorch.losses import JaccardLoss

def get_loss_fn(loss_args):
    loss_fn = loss_args.get("loss_fn")
    print(f"Loss function specified: {loss_fn}")  # Debugging line
    if loss_fn == "BCE":
        return torch.nn.BCEWithLogitsLoss()
    elif loss_fn == "CE":
        return torch.nn.CrossEntropyLoss()
    elif loss_fn == "Dice":
        return DiceLoss()
    elif loss_fn == "IoU":
        return JaccardLoss(mode='multiclass')
    else:
        raise ValueError(f"loss_fn {loss_fn} not supported.")