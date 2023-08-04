# %%
from tkinter import Text
import torch
import torchvision
import psycopg2
import dotenv
import httpx
import PIL
import os
import re


dotenv.load_dotenv("/root/tinier-world/.env")
IMAGE_S3 = "https://d1fgjcgtpkti4f.cloudfront.net/{KEY}.jpg"
TEXT_S3 = "https://dxcdooe3ky7rd.cloudfront.net/{KEY}.txt"
IMAGE_QUERY = """
  WITH numbered_rows AS (
    SELECT
      item_key,
      label,
      ROW_NUMBER() OVER (PARTITION BY label ORDER BY item_key) AS rn
    FROM img_labels
    ORDER BY
      item_key
  )
  SELECT
    item_key,
    label
  FROM numbered_rows
  WHERE
    rn <= 100;
"""
TEXT_QUERY = """
  SELECT
    item_key,
    ROW_NUMBER() OVER (ORDER BY item_key) AS rn
  FROM items
  WHERE type = 'txt'
  ORDER BY item_key
  LIMIT 100;
"""


# %%
class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, params):
        self.conn = psycopg2.connect(**params)
        self.cursor = self.conn.cursor()
        self.cache_dir = "/root/tinier-world/images"
        self.transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(256),
                torchvision.transforms.CenterCrop(224),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        self.item_keys, self.labels = self.load_data_from_database()
        self.label_dict = {label: i for i, label in enumerate(sorted(set(self.labels)))}
        self.labels = [self.label_dict[label] for label in self.labels]
        self.labels = torch.tensor(self.labels)

    def load_data_from_database(self):
        self.cursor.execute(IMAGE_QUERY)
        rows = self.cursor.fetchall()
        item_keys, labels = zip(*rows)
        return item_keys, labels

    def __len__(self):
        return len(self.item_keys)

    def __getitem__(self, index):
        item_key = self.item_keys[index]
        label = self.labels[index]
        item_path = os.path.join(self.cache_dir, f"{item_key}.jpg")
        if not os.path.exists(item_path):
            s3_url = IMAGE_S3.format(KEY=item_key)
            res = httpx.get(s3_url)
            os.makedirs(self.cache_dir, exist_ok=True)
            f = open(f"{self.cache_dir}/{item_key}.jpg", "wb")
            f.write(res.content)
            f.close()

        image = PIL.Image.open(item_path)
        image = self.transform(image)
        return image, label


# %%
class TextDataset(torch.utils.data.Dataset):
    def __init__(self, params):
        self.conn = psycopg2.connect(**params)
        self.cursor = self.conn.cursor()
        self.cache_dir = "/root/tinier-world/text"

        self.item_keys = self.load_data_from_database()

    def load_data_from_database(self):
        self.cursor.execute(TEXT_QUERY)
        rows = self.cursor.fetchall()
        item_keys, _ = zip(*rows)
        return item_keys

    def __len__(self):
        return len(self.item_keys)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        item_key = self.item_keys[index]
        item_path = os.path.join(self.cache_dir, f"{item_key}.txt")
        if os.path.exists(item_path):
            with open(item_path, "r", encoding="utf-8") as file:
                text = file.read()
                return text

        s3_url = TEXT_S3.format(KEY=item_key)
        res = httpx.get(s3_url)
        text = res.text
        with open(item_path, "w", encoding="utf-8") as file:
            file.write(text)
        return text
