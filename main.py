import os
import fire
from pytorch_lightning import Trainer
import glob

from util import init_exp_folder, Args
from lightning import (get_task,
                       load_task,
                       get_ckpt_callback, 
                       get_early_stop_callback,
                       get_logger)

# Model inspection debug
import torch
from models import get_model

def train(save_dir="../archive/results",  # TODO ~ Customize
          exp_name="segdemo",  # TODO ~ Customize
          model="densenet121",  # TODO ~ Customize
          task='classification',  # TODO ~ Customize
          devices=1,  # Replaces gpus; number of devices to use
          accelerator="gpu",  # Use "gpu" or "cpu"
          pretrained="imagenet",
          num_classes=1,  # TODO ~ Customize
          logger_type='wandb',
          gradient_clip_val=0.5,
          max_epochs=200,
          patience=10,
          stochastic_weight_avg=True,
          limit_train_batches=16.0,
          tb_path="./tb",
          loss_fn="BCE", # TODO ~ Customize (Dice for Segmentation)
          weights_summary=None,
          proj_name="rad_noise",
          dataset_path="../chestxray-8/chestxray-8-labels.csv",  # TODO ~ Customize
          quantum=0.0,
          electronic=0.0,
          ):
    """
    Run the training experiment.
    """
    # Add model_args here
    num_channels = 3 if task == "classification" else 1
    model_args = {
        "task": task,
        "model": model,
        "encoder": "resnet34",  # Fixed for this task
        "pretrained": pretrained,
        "num_classes": num_classes,
        "num_channels": num_channels,  # Default to 3 for RGB
        "loss_fn": loss_fn,
        "dataset_path": dataset_path,
        "quantum": quantum,
        "electronic": electronic,
    }

    args = Args(locals())
    args["model_args"] = model_args  # Explicitly include model_args in Args
    for key, value in args.items():
      print(f"{key}: {value}") 
    init_exp_folder(args)
    task = get_task(args)
    trainer = Trainer(devices=devices,
                      accelerator=accelerator,
                      logger=get_logger(logger_type, save_dir, exp_name, proj_name),
                      callbacks=[get_early_stop_callback(patience),
                                 get_ckpt_callback(save_dir, exp_name)],
                      gradient_clip_val=gradient_clip_val,
                      limit_train_batches=limit_train_batches,
                      max_epochs=max_epochs)
    trainer.fit(task)


def test(exp_name,
         gpus=4,
         **kwargs):
    """
    Run the testing experiment.

    Args:
        ckpt_path: Path for the experiment to load
        gpus: int. (ie: 2 gpus)
             OR list to specify which GPUs [0, 1] OR '0,1'
             OR '-1' / -1 to use all available gpus
    Returns: None

    """
    ckpt_dir = os.path.join("../archive/results/", exp_name)
    ckpt_path = glob.glob(os.path.join(ckpt_dir, "ckpts/*.ckpt"))

    if not ckpt_path:
        raise ValueError(f"No checkpoint found in {ckpt_dir}")
    ckpt_path = ckpt_path[0]
    task = load_task(ckpt_path, **kwargs)
    trainer = Trainer(devices=gpus)
    trainer.test(task)

sample_model_args = {
    "task": "classification",
    "model": "densenet121",
    "encoder": "resnet34",
    "pretrained": "imagenet",
    "num_classes": 1,
    "num_channels": 1,
    "loss_fn": "BCE",
    "dataset_path": "../chestxray-8/chestxray-8-labels.csv"
}

def inspect_model(ckpt_path:str, args_dict = sample_model_args):
    model = get_model(args_dict)
    checkpoint = torch.load(ckpt_path)
    state_dict = checkpoint['state_dict']
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key[len('model.'): ]  # Remove the "model." prefix
        new_state_dict[new_key] = value

    model.load_state_dict(new_state_dict, strict=False)
    result = model.load_state_dict(new_state_dict, strict=False)

    missing_keys = result.missing_keys
    unexpected_keys = result.unexpected_keys

    print(f"Number of missing keys: {len(missing_keys)}")
    print(f"Number of unexpected keys: {len(unexpected_keys)}")
    breakpoint() 

    model.eval()


if __name__ == "__main__":
    fire.Fire()
