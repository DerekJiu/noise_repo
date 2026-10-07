import pytorch_lightning as pl
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
import numpy as np


from models import get_model
from data import SegmentationDataset #TODO ~ redefine
from .logger import TFLogger
from eval import get_loss_fn, SegmentationEvaluator
from eval.segmentation import get_segmentation_metrics, get_optimal_f1

class SegmentationTask(pl.LightningModule, TFLogger):
    def __init__(self, params):
        super().__init__()
        self.save_hyperparameters(params)
        self.params = params
        self.dataset_path = self.params["dataset_path"]
        self.model = get_model(self.params)
        self.loss_fn = get_loss_fn(self.params)
        self.evaluator = SegmentationEvaluator(0.5)
        self.validation_outputs = []
        self.iou = []
        self.quantum_intensity = self.params.get('quantum')
        self.electronic_intensity = self.params.get('electronic')

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_nb):
        images, masks = batch['image'], batch['mask']
        preds = self.model(images)
        preds = torch.sigmoid(preds)
        loss = self.loss_fn(preds, masks)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_nb):
        images, masks = batch['image'], batch['mask']
        preds = self.model(images)  # preds: (batch_size, num_classes, height, width)

        if preds.dim() == 4:
            raw_preds = torch.sigmoid(preds)
            b, c, h, w = raw_preds.shape
            eval_preds = raw_preds.view(b, c*h*w)
            
        if masks.dim() == 4: #masks are already binarized
            b2, c2, h2, w2 = masks.shape
            eval_masks = masks.view(b2, c2*h2*w2)
        
        iou_preds = torch.sigmoid(preds)
        iou_preds = iou_preds.cpu().numpy()
        iou_labels = masks.cpu().numpy()
        _, threshold = get_optimal_f1(iou_labels, iou_preds, return_threshold=True)
        iou_preds = (iou_preds > threshold).astype(int)
        intersection = np.logical_and(iou_labels, iou_preds).sum()
        union = np.logical_or(iou_labels, iou_preds).sum()
        iou = intersection/union

        self.evaluator.update((eval_preds, eval_masks))
        loss = self.loss_fn(raw_preds, masks)
        self.validation_outputs.append(loss)
        self.log("val_loss_step", loss, prog_bar=False)
        
        eval_preds = (eval_preds.cpu().numpy() > 0.5).astype(int)
        
        intersection = np.logical_and(eval_masks.cpu().numpy(), eval_preds).sum()
        union = np.logical_or(eval_masks.cpu().numpy(), eval_preds).sum()
        iou = intersection/union
        self.log("IOU_step", iou, prog_bar=False)
        self.iou.append(iou)

        return loss

    def on_validation_epoch_end(self):
        avg_loss = torch.stack(self.validation_outputs).mean()
        self.log("val_loss", avg_loss, prog_bar=True)
        avg_iou = sum(self.iou) / len(self.iou)
        self.log("iou_real", avg_iou, prog_bar=False)
        metrics = self.evaluator.evaluate()
        self.evaluator.reset()
        self.log_dict(metrics, prog_bar=True)
        self.iou.clear()
        self.validation_outputs.clear()
        

    def test_step(self, batch, batch_nb):
        return self.validation_step(batch, batch_nb)

    def on_test_epoch_end(self):
        return self.on_validation_epoch_end()

    def configure_optimizers(self):
        return [torch.optim.Adam(self.parameters(), lr=0.02)]
    '''
    def collate_fn(self, batch):
        return {
            "images": torch.stack([item["image"] for item in batch]),
            "masks": torch.stack([item["mask"] for item in batch]),
            "image_paths": [item["image_path"] for item in batch]
        }
    '''

    def train_dataloader(self):
        dataset = SegmentationDataset(
            dataset_path=self.dataset_path,
            split="train",
            transforms=None,
            quantum=self.quantum_intensity,
            electronic=self.electronic_intensity
        )
        print(f"Train dataset length: {len(dataset)}")
        return DataLoader(dataset, batch_size=2, shuffle=True, num_workers=8)#, collate_fn=self.collate_fn)

    def val_dataloader(self):
        dataset = SegmentationDataset(
            dataset_path=self.dataset_path,
            split="val",
            transforms=None,
            quantum=self.quantum_intensity,
            electronic=self.electronic_intensity
        )
        print(f"Val dataset length: {len(dataset)}")
        return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8)#, collate_fn=self.collate_fn)

    def test_dataloader(self):
        dataset = SegmentationDataset(
            dataset_path=self.dataset_path,
            split="test",
            transforms=None,
        )
        print(f"Test dataset length: {len(dataset)}")
        return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8)#collate_fn=self.collate_fn)
