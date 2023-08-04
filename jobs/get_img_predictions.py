#%%
import torch
import torchvision
import psycopg2
import dotenv
import tqdm
import httpx
import json
import PIL
import io
import os

#%%
dotenv.load_dotenv("/root/tinier-world/.env")
USER = os.environ.get("POSTGRES_USER")
DATABASE = os.environ.get("POSTGRES_DB")
PASSWORD = os.environ.get("POSTGRES_PASSWORD")
p = { "host": "localhost", "user": USER, "port": 5432, "database": DATABASE, "password": PASSWORD }
S3 = "https://d1fgjcgtpkti4f.cloudfront.net/{KEY}.jpg"

#%%
class ProductionDataset(torch.utils.data.Dataset):
  def __init__(self, params):
    self.params = params
    self.transform = torchvision.transforms.Compose([
      torchvision.transforms.Resize(256),
      torchvision.transforms.CenterCrop(224),
      torchvision.transforms.ToTensor(),
      torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

  def execute_query(self, query, args=()):
    with psycopg2.connect(**self.params) as conn:
      conn.autocommit = True
      with conn.cursor() as cursor:
        cursor.execute(query, args)
        return cursor.fetchone()

  def __len__(self):
    num, = self.execute_query("SELECT COUNT(*) FROM items WHERE type = 'img'")
    return num

  def __getitem__(self, index):
    item_key, = self.execute_query("SELECT item_key FROM items WHERE type = 'img' ORDER BY item_key OFFSET %s LIMIT 1", (index,))
    cache_dir = "/root/tinier-world/images"
    item_path = os.path.join(cache_dir, f"{item_key}.jpg")
    if os.path.exists(item_path):
      image = PIL.Image.open(item_path)
      image = self.transform(image)
      return image, item_key
    print(f"downloading {item_key}")
    s3_url = S3.format(KEY=item_key)
    res = httpx.get(s3_url)
    image = PIL.Image.open(io.BytesIO(res.content))
    image = self.transform(image)
    return image, item_key

#%%
class CustomNet(torch.nn.Module):
  def __init__(self, num_classes):
    super(CustomNet, self).__init__()
    self.model = torch.hub.load("pytorch/vision:v0.10.0", "resnet101", weights="ResNet101_Weights.IMAGENET1K_V1")
    self.model.fc = torch.nn.Linear(2048, 512, bias=True)
    self.relu = torch.nn.ReLU()
    self.last = torch.nn.Linear(512, num_classes, bias=True)

  def forward(self, x):
    x = self.model(x)
    # save embeddings
    x = self.relu(x); emb = x
    x = self.last(x)
    return x, emb
#%%
ds = ProductionDataset(p)
dl = torch.utils.data.DataLoader(ds, batch_size=16, num_workers=12)
with open('class_map.json', 'r') as fp: class_map = json.load(fp)
model = CustomNet(24)
#%%
model.load_state_dict(torch.load("/root/tinier-world/weights/00.pth"))
for param in model.parameters(): param.requires_grad = False
model.eval()
#%%
conn = psycopg2.connect(**p)
conn.autocommit = True
cursor = conn.cursor()
cursor.execute("""
  BEGIN;

  CREATE TABLE IF NOT EXISTS img_predictions (
    item_key   UUID        PRIMARY KEY,
    created_at TIMESTAMP   NOT NULL DEFAULT NOW(),
    prediction VARCHAR(20) NOT NULL,
    model_name VARCHAR(20) NOT NULL,
    version    VARCHAR(20) NOT NULL DEFAULT 'v0',
    embedding  FLOAT[]     NOT NULL
  );

  COMMIT;
""")

INSERT_QUERY = """
  INSERT INTO img_predictions (item_key, prediction, model_name, embedding)
  VALUES (%s, %s, %s, %s) ON CONFLICT (item_key) DO NOTHING;
"""
#%%
for batch, (images, keys) in tqdm.tqdm(enumerate(dl), total=len(dl)):
  logits, embeddings = model(images)
  _, preds = torch.max(logits, 1)
  for i, key in enumerate(keys):
    pred = class_map[str(preds[i].item())]
    embedding = embeddings[i].tolist()
    cursor.execute(INSERT_QUERY, (key, pred, "resnet101", embedding))