import numpy as np
import torch
from PIL import Image
import torchvision.transforms as T
import pandas as pd 
from torchvision.transforms import ToTensor
from .noise import add_quantum, add_electronic


class GeneralizedClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, split, transforms=None, quantum=0, electronic=0):
        df = pd.read_csv(dataset_path)
        self.dataset = df.loc[df['split'] == split].reset_index(drop=True) 
        self.transforms = transforms 
        self.quantum_intensity = quantum
        self.electronic_intensity = electronic

    def __len__(self):
        return len(self.dataset) 
    
    def __getitem__(self, index):
        sample = self.dataset.loc[index]
        image = Image.open(sample["image_path"]).convert('L')
        
        label = sample["label"]

        if self.quantum_intensity > 0:
            image = add_quantum(image, intensity=self.quantum_intensity)

        if self.electronic_intensity > 0:
            image = add_electronic(image, intensity=self.electronic_intensity)

        image = image.convert('L')
        
        if self.transforms:
            image = self.transforms(image)
        return {"image": image, "label": label}


class ImageClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, image_path=None, labels=None, transforms=None):
        self._image_path = image_path
        self._labels = labels
        self._transforms = transforms

    def __len__(self):
        return len(self._labels)

    def __getitem__(self, index):
        label = torch.tensor(np.float64(self._labels[index]))
        image = Image.open(self._image_path[index]).convert('RGB')
        if self._transforms is not None:
            image = self._transforms(image)

        return image, label

#TODO ~ Complete definition 
class SegmentationDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, split="train", transforms=None, target_size=(512, 512), quantum=0, electronic=0):
        self.data = pd.read_csv(dataset_path)
        self.data = self.data[self.data['split'] == split].reset_index(drop=True)
        self.transforms = transforms
        self.target_size = target_size
        self.quantum_intensity = quantum
        self.electronic_intensity = electronic

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data.iloc[index]
        image_path = sample['image_path']
        mask_path = sample['mask_path']

        # Load image and mask
        image = Image.open(image_path).convert('L')
        mask = Image.open(mask_path).convert('L')

        # Resize to target size
        image = image.resize(self.target_size)
        mask = mask.resize(self.target_size, resample=Image.NEAREST)

        if self.quantum_intensity > 0:
            image = add_quantum(image, intensity=self.quantum_intensity)
            print("Quantum noise being applied")

        if self.electronic_intensity > 0:
            image = add_electronic(image, intensity=self.electronic_intensity)
            print("Electronic noise being applied")
            
        
        # Convert to tensors
        image = image.convert('L')
        image = T.ToTensor()(image)
        mask = T.ToTensor()(mask)
        mask = (mask > 0.5).float()

        return {"image": image, "mask": mask, "image_path": image_path}
