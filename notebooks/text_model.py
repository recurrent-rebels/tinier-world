# %%
import torch
import random
import numpy
import torch.nn as nn
import dataset
from transformers import BertTokenizer, BertModel
import os
import random
import time

# %%
torch.manual_seed(42)
numpy.random.seed(42)
random.seed(42)
gen = torch.Generator().manual_seed(42)


# %%
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32 + worker_id
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)


# %%
USER = os.environ.get("POSTGRES_USER")
DATABASE = os.environ.get("POSTGRES_DB")
PASSWORD = os.environ.get("POSTGRES_PASSWORD")
params = {
    "host": "localhost",
    "user": USER,
    "port": 5432,
    "database": DATABASE,
    "password": PASSWORD,
}
# %%
ds = dataset.TextDataset(params)


# %%
class CustomModel(torch.nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        self.model_name = "bert-base-cased"
        self.model = BertModel.from_pretrained(self.model_name)
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.projection_layer = nn.Linear(768, 512)

    def forward(self, **inputs):
        outputs = self.model(**inputs)
        # save embeddings
        x = outputs.last_hidden_state
        emb = self.projection_layer(x)
        return emb


# %%
model = CustomModel()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model = model.to(device)

embeddings = []

texts = ds[0:5]
texts_inputs = model.tokenizer(texts, padding=True, return_tensors="pt")
texts_embeddings = []

start_time = time.time()

with torch.no_grad():
    outputs = model(**texts_inputs)
    text_embeddings = outputs.mean(dim=1)

end_time = time.time()
print("Time for batch processing: ", end_time - start_time)

start_time = time.time()
for text in texts:
    inputs = model.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        embedding = outputs.mean(dim=1)
        embeddings.append(embedding)
end_time = time.time()
print("Time for single processing: ", end_time - start_time)
embeddings = torch.cat(embeddings)

# %%
index = random.randint(0, 4)
embedding = embeddings[index]
embedding_2 = text_embeddings[index]

similarities = torch.cosine_similarity(embedding.unsqueeze(0), embeddings, dim=1)
similarities_2 = torch.cosine_similarity(
    embedding_2.unsqueeze(0), text_embeddings, dim=1
)

similarities[index] = -float("inf")
similarities_2[index] = -float("inf")

most_similar_index = torch.argmax(similarities)
most_similar_index_2 = torch.argmax(similarities_2)
# %%
print(f"{index} is most similar to {most_similar_index}")
print(most_similar_index_2)
# %%

print(ds[index])
print(ds[most_similar_index])
print(ds[most_similar_index_2])
# %%
