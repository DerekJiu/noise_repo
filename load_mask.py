import argparse
import os
import random
import csv
import numpy as np
import torch
import torchvision.transforms as transforms
from pathlib import Path
from typing import Union, Tuple, Optional
from tqdm import tqdm
import torch.backends.cudnn as cudnn
from PIL import Image, ImageDraw, ImageFont
import logging
import glob
from torchvision.transforms import ToPILImage
from lightning import SegmentationTask  # Ensure you have the correct import
import matplotlib.pyplot as plt


def load_model(exp_name, device):
    """
    Load and initialize the segmentation model from a checkpoint.

    Args:
        exp_name (str): Experiment name to locate the model checkpoint.
        device (str): Device to load the model on ('cuda:0', 'cpu', etc.).

    Returns:
        SegmentationTask: Loaded and initialized model in evaluation mode.
    """
    ckpt_path = f'../archive/results/{exp_name}/ckpts/*.ckpt'
    checkpoint_files = glob.glob(ckpt_path)
    print("KSFJLKSDFJKLSDF", glob.glob(f'../archive/results/{exp_name}/ckpts/*.ckpt'))

    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint found at {ckpt_path}")
    checkpoint_path = checkpoint_files[0]
    model = SegmentationTask.load_from_checkpoint(checkpoint_path, map_location=device, strict=False)
    model.to(device)
    model.eval()
    print(f"Loaded checkpoint from: {checkpoint_path}")
    return model

def predict(model, image_tensor, device):
    """
    Perform model inference on a preprocessed image tensor and return the segmentation mask.

    Args:
        model (SegmentationTask): The loaded segmentation model.
        image_tensor (torch.Tensor): Preprocessed image tensor of shape (1, 3, height, width).
        device (str): Device to perform inference on ('cuda:0', 'cpu', etc.).

    Returns:
        np.ndarray: The predicted segmentation mask as a NumPy array.
    """
    image_tensor = image_tensor.to(device)
    print("Image tensor values:", image_tensor)
    print("Image tensor shape:", image_tensor.shape)  # Should be (1, 3, H, W)
    print("Image tensor min/max:", image_tensor.min().item(), image_tensor.max().item())
    with torch.no_grad():
        # Get segmentation mask output from the model
        output = model(image_tensor)
        for i in range(output.shape[1]):  # Loop over classes
        class_probs = probabilities[0, i].cpu().numpy()  # Get probability map for class i
        print(f"Class {i} probability min/max:", class_probs.min(), class_probs.max())
    
        plt.imshow(class_probs, cmap="jet")
        plt.colorbar()
        plt.title(f"Class {i} Probability Map")
        plt.savefig(f"class_{i}_prob_map.png")

        
        print(f"Model output shape: {output.shape}")  # Debugging output
        
        print("Output min/max:", output.min().item(), output.max().item())
        print("Raw output values (flattened):")
        print(output.cpu().numpy().flatten())
        
        # Convert the model output to a mask by taking the argmax along the class dimension
       # probabilities = torch.nn.functional.softmax(output, dim=1)
        #pobabilities = torch.nn.functional.softmax(output, dim=1)  # Normalize across classes
        probabilities = torch.nn.functional.softmax(output, dim=1)
        mask = torch.argmax(probabilities, dim=1).squeeze(0).cpu().numpy()

        print("Mask array:")
        print(mask)
        
        plt.imshow(mask, cmap='jet')
        plt.colorbar()
        plt.savefig('output_colored_mask.png')
        # Debug: Print unique values in the predicted mask
        print("Unique values in mask:", np.unique(mask))
        
        # Convert the mask to a PIL image for visualization
        # Here we assume a binary mask: multiply by 255 to convert class 1 to white
        mask_image = Image.fromarray((mask * 255).astype(np.uint8))
        mask_image.save('output_mask.png')
    
    return mask

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, required=True, help='Experiment name for checkpoint retrieval')
    parser.add_argument('--image_path', type=str, required=True, help='Path to the image for inference')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use for inference')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Use GPU if available, otherwise fallback to CPU
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = load_model(args.exp_name, device)
    logger.info("Model loaded successfully")
    
    # Define your preprocessing transform.
    # IMPORTANT: Modify the normalization values if your training pipeline used different ones.
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    # Open the input image and convert to RGB
    image = Image.open(args.image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)
    
    # Run prediction to get the segmentation mask
    mask = predict(model, image_tensor, device)
    logger.info("Segmentation mask saved as output_mask.png")

if __name__ == "__main__":
    main()
