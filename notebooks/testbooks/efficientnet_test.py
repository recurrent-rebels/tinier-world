#%%
import torch
import random
import numpy
import dataset
import wandb
import utils
import json
import os
from efficientnet_pytorch import EfficientNet
#%%
torch.manual_seed(42)
numpy.random.seed(42)
random.seed(42)
gen = torch.Generator().manual_seed(42)
#%%
def seed_worker(worker_id):
  worker_seed = torch.initial_seed() % 2**32 + worker_id
  numpy.random.seed(worker_seed)
  random.seed(worker_seed)
#%%
USER = os.environ.get("POSTGRES_USER")
DATABASE = os.environ.get("POSTGRES_DB")
PASSWORD = os.environ.get("POSTGRES_PASSWORD")
params = { "host": "localhost", "user": USER, "port": 5432, "database": DATABASE, "password": PASSWORD }
#%%
ds = dataset.ImageDataset(params)
reverse_label_dict = {value: key for key, value in ds.label_dict.items()}
with open('class_map.json', 'w') as fp: json.dump(reverse_label_dict, fp)
trn_len = int(0.8 * len(ds))
tst_len = len(ds) - trn_len
trn_ds, tst_ds = torch.utils.data.random_split(ds, [trn_len, tst_len], generator=gen)
trn_dl = torch.utils.data.DataLoader(trn_ds, batch_size=8, shuffle=True, num_workers=4, worker_init_fn=seed_worker, generator=gen)
tst_dl = torch.utils.data.DataLoader(tst_ds, batch_size=8, shuffle=True, num_workers=4, worker_init_fn=seed_worker, generator=gen)
#%%
class CustomModel(torch.nn.Module):
  def __init__(self, num_classes):
    super(CustomModel, self).__init__()
    self.model = EfficientNet.from_pretrained('efficientnet-b7')
    self.model.fc = torch.nn.Linear(2560, 1000, bias=True)
    self.relu = torch.nn.ReLU()
    self.last = torch.nn.Linear(1000, num_classes, bias=True)

  def forward(self, x):
    x = self.model(x)
    # save embeddings
    x = self.relu(x); emb = x
    x = self.last(x)
    return x, emb
#%%
model = CustomModel(24)
print(model)
for param in model.parameters(): param.requires_grad = False
for param in model.model.fc.parameters(): param.requires_grad = True
for param in model.last.parameters(): param.requires_grad = True

# check if parameters are freeze
#double check which parameters is trainable
for name, param in model.named_parameters():
   if param.requires_grad:
       print(name)
#%%
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(list(model.model.fc.parameters()) + list(model.last.parameters()), lr=0.001)
#%%
config = { "learning_rate": 0.001, "epochs": 2 }
wandb.init(project="tiny-imgs-classification", config=config)
#%%
for epoch in range(3):
  model.train()
  trn_loss = 0.0
  trn_acc  = 0.0
  for i, (x, y) in enumerate(trn_dl):
    # utils.show_batch_images(x, y.tolist(), reverse_label_dict)
    optimizer.zero_grad()
    output, emb = model(x)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()
    trn_loss += loss.item()
    trn_acc  += utils.accuracy(output, y)

    if (i % 50 == 0) and (i > 0):
      val_loss = 0.0
      val_acc  = 0.0
      with torch.no_grad():
        model.eval()
        for _, (x, y) in enumerate(tst_dl):
          output, emb = model(x)
          loss = criterion(output, y)
          val_loss += criterion(output, y).item()
          val_acc  += utils.accuracy(output, y)
      val_loss /= len(tst_dl)
      val_acc  /= len(tst_dl)
      trn_loss /= 50
      trn_acc  /= 50
      wandb.log({"val_acc": val_acc, "val_loss": val_loss, "trn_acc": trn_acc, "trn_loss": trn_loss})
      print(f"Epoch {epoch} | Batch {i:4} | trn_loss {trn_loss:.4f} | trn_acc {trn_acc:.4f} | val_loss {val_loss:.4f} | val_acc {val_acc:.4f}")


#%%
torch.save(model.state_dict(), "/root/tinier-world/weights/00.pth")