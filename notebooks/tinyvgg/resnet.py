#%%
from pathlib import Path
import os
import random

import wandb
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch import nn
import torchvision.models as models
import numpy as np

import walk_through_directory
import train_utils

from tqdm.auto import tqdm

# %%
device = "cuda" if torch.cuda.is_available() else "cpu"
data_path = Path("../../dataset/")
train_path = data_path / "train"
test_path = data_path / "test"

# %%
# Check what we have in our dataset.
walk_through_directory.walk_through_dir(data_path)

#%%
# *** What is this doing?
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
gen = torch.Generator().manual_seed(42)
#%%
# *** What is the purpose of this function?
def seed_worker(worker_id):
  worker_seed = torch.initial_seed() % 2**32 + worker_id
  np.random.seed(worker_seed)
  random.seed(worker_seed)

# %%
# Create our transform function, train datase and test dataset.
transform = transforms.Compose([
      transforms.Resize(256),
      transforms.CenterCrop(224),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

train_ds = datasets.ImageFolder(root=train_path, 
                                  transform=transform, 
                                  target_transform=None)

test_ds = datasets.ImageFolder(root=test_path, 
                                 transform=transform)
# %%
# Check how many CPUs are available on the machine.
num_workers = os.cpu_count()
num_workers

# %%
train_dl = DataLoader(train_ds, 
                      batch_size=8, 
                      shuffle=True, 
                      num_workers=num_workers, 
                      worker_init_fn=seed_worker, 
                      generator=gen)
# Why are we shuffling the training data?
test_dl = DataLoader(test_ds, 
                     batch_size=8, 
                     shuffle=True, 
                     num_workers=num_workers,
                     worker_init_fn=seed_worker, 
                     generator=gen)

# %%
# Get class names as a list, check how many classes there are.
class_names = train_ds.classes
print(f"Number of classes: {len(class_names)}\nClass names: {class_names}")
# %%
# Check the lengths of datasets.
len(train_ds), len(test_ds)

# %%
# Load the ResNet18 model from torchvision with pre-trained weights
# model = models.resnet18(pretrained="ResNet18_Weights.DEFAULT")

# %%
# Use Bes Custom model
class CustomResNet(torch.nn.Module):
  def __init__(self, num_classes):
    super(CustomResNet, self).__init__()
    self.resnet = torch.hub.load("pytorch/vision:v0.10.0", "resnet18", weights="ResNet18_Weights.IMAGENET1K_V1")
    self.resnet.fc = torch.nn.Linear(512, 256, bias=True)
    self.relu = torch.nn.ReLU()
    self.last = torch.nn.Linear(256, num_classes, bias=True)

  def forward(self, x):
    x = self.resnet(x)
    x = self.relu(x)
    # save embeddings
    emb = x
    x = self.last(x)
    return x
#%%
model = CustomResNet(24)
for param in model.parameters(): param.requires_grad = False
for param in model.resnet.fc.parameters(): param.requires_grad = True
for param in model.last.parameters(): param.requires_grad = True
#%%
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
NUM_EPOCHS = 3
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(list(model.resnet.fc.parameters()) + list(model.last.parameters()), lr=0.001)

# %%
# Start the timer
from timeit import default_timer as timer 
start_time = timer()

# Train model_1
model_results = train_utils.train(model=model, 
                        train_dataloader=train_dl,
                        test_dataloader=test_dl,
                        optimizer=optimizer,
                        loss_fn=criterion, 
                        epochs=NUM_EPOCHS)

# End the timer and print out how long it took
end_time = timer()
print(f"Total training time: {end_time-start_time:.3f} seconds")

# %%
print(model_results)

# %%
import matplotlib.pyplot as plt

def plot_loss_curves(results):
    """Plots training curves of a results dictionary.

    Args:
        results (dict): dictionary containing list of values, e.g.
            {"train_loss": [...],
             "train_acc": [...],
             "test_loss": [...],
             "test_acc": [...]}
    """
    
    # Get the loss values of the results dictionary (training and test)
    loss = results['train_loss']
    test_loss = results['test_loss']

    # Get the accuracy values of the results dictionary (training and test)
    accuracy = results['train_acc']
    test_accuracy = results['test_acc']

    # Figure out how many epochs there were
    epochs = range(len(results['train_loss']))

    # Setup a plot 
    plt.figure(figsize=(15, 7))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label='train_loss')
    plt.plot(epochs, test_loss, label='test_loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label='train_accuracy')
    plt.plot(epochs, test_accuracy, label='test_accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.legend()
# %%
plot_loss_curves(model_results)

# %%
