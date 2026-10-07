import torch

class ClassificationDebugger():

    def inspect(self, logits, y, loss):
        print("==== DEBUG INFO ====")
        
        print(f"Logits Shape: {logits.shape}")
        print(f"Logits Sample: {logits[:5]}")
        
        print(f"Labels Shape: {y.shape}, dtype: {y.dtype}")
        print(f"Labels Sample: {y[:5]}")
        
        print(f"Loss Shape: {loss.shape}, Value: {loss.item()}")
        
        print(f"Sigmoid(Logits) Sample: {torch.sigmoid(logits)[:5]}")
        
        print(f"Logits Type: {type(logits)}, Requires Grad: {logits.requires_grad}")
        print(f"Labels Type: {type(y)}, Requires Grad: {y.requires_grad if y.requires_grad else 'N/A'}")
        
        print("====================")

class SegmentationDebugger():

    def inspect(self, preds, masks, loss):
        print("==== DEBUG INFO ====")
        
        print(f"Preds Shape: {preds.shape}")
        print(f"Preds Sample: {preds[:5]}")
        
        print(f"Masks Shape: {masks.shape}, dtype: {masks.dtype}")
        print(f"Masks Sample: {masks[:5]}")
        
        print(f"Loss Shape: {loss.shape}, Value: {loss.item()}")
        
        print(f"Sigmoid(Preds) Sample: {torch.sigmoid(preds)[:5]}")
        
        print(f"Preds Type: {type(preds)}, Requires Grad: {preds.requires_grad}")
        print(f"Masks Type: {type(masks)}, Requires Grad: {masks.requires_grad if masks.requires_grad else 'N/A'}")
        
        print("====================")