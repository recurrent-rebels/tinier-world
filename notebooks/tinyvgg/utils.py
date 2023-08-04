
import torch

def accuracy(output, target):
  with torch.inference_mode():
    pred = torch.argmax(output, dim=1)
    correct = pred.eq(target)
    acc = correct.float().mean().item()
    return acc