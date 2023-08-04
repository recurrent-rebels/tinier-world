# %%
import torch
import torchvision
import torch.nn as nn
from transformers import XLNetTokenizer, XLNetModel
import psycopg2
import dotenv
import tqdm
import httpx
import json
import PIL
import io
import os

# %%
dotenv.load_dotenv("/root/tinier-world/.env")
USER = os.environ.get("POSTGRES_USER")
DATABASE = os.environ.get("POSTGRES_DB")
PASSWORD = os.environ.get("POSTGRES_PASSWORD")
p = {
    "host": "localhost",
    "user": USER,
    "port": 5432,
    "database": DATABASE,
    "password": PASSWORD,
}
S3 = "https://dxcdooe3ky7rd.cloudfront.net/{KEY}.txt"


# %%
class ProductionDataset(torch.utils.data.Dataset):
    def __init__(self, params):
        self.params = params

    def execute_query(self, query, args=()):
        with psycopg2.connect(**self.params) as conn:
            conn.autocommit = True
            with conn.cursor() as cursor:
                cursor.execute(query, args)
                return cursor.fetchone()

    def __len__(self):
        (num,) = self.execute_query("SELECT COUNT(*) FROM items WHERE type = 'txt'")
        return num

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        (item_key,) = self.execute_query(
            "SELECT item_key FROM items WHERE type = 'txt' ORDER BY item_key OFFSET %s LIMIT 1",
            (index,),
        )
        cache_dir = "/root/tinier-world/text"
        item_path = os.path.join(cache_dir, f"{item_key}.txt")
        if os.path.exists(item_path):
            with open(item_path, "r", encoding="utf-8") as file:
                text = file.read()
                return text, item_key

        print(f"downloading {item_key}")
        s3_url = S3.format(KEY=item_key)
        res = httpx.get(s3_url)
        text = res.text
        with open(item_path, "w", encoding="utf-8") as file:
            file.write(text)
        return text, item_key


# %%
class CustomModel(torch.nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        self.model_name = "xlnet-base-cased"
        self.model = XLNetModel.from_pretrained(self.model_name)
        self.tokenizer = XLNetTokenizer.from_pretrained(self.model_name)
        self.projection_layer = nn.Linear(768, 512)

    def forward(self, **inputs):
        outputs = self.model(**inputs)
        # save embeddings
        x = outputs.last_hidden_state
        emb = self.projection_layer(x)
        return emb


# %%
ds = ProductionDataset(p)
dl = torch.utils.data.DataLoader(ds, batch_size=16, num_workers=12)
model = CustomModel()
# %%
model.eval()
# %%
conn = psycopg2.connect(**p)
conn.autocommit = True
cursor = conn.cursor()
cursor.execute(
    """
  BEGIN;

  CREATE TABLE IF NOT EXISTS txt_predictions (
    item_key   UUID        PRIMARY KEY,
    created_at TIMESTAMP   NOT NULL DEFAULT NOW(),
    model_name VARCHAR(20) NOT NULL,
    version    VARCHAR(20) NOT NULL DEFAULT 'v0',
    embedding  FLOAT[]     NOT NULL
  );

  COMMIT;
"""
)

INSERT_QUERY = """
  INSERT INTO txt_predictions (item_key, model_name, embedding)
  VALUES (%s, %s, %s) ON CONFLICT (item_key) DO NOTHING;
"""
# %%
for batch, (texts, keys) in tqdm.tqdm(enumerate(dl), total=len(dl)):
    for text, key in zip(texts, keys):
        inputs = model.tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            embedding = outputs.mean(dim=1)
            cursor.execute(INSERT_QUERY, (key, "XLNet", embedding.tolist()))

# %%
print(len(dl))
# %%
