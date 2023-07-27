import faiss
import pickle
import numpy as np
import redis

# user id # recommended_items_id 
# /home -> kafka -> workers -> postgres
# postgres (item_id, user_id, embeddings) -> workers (faiss -> recommended items) -> update redis
redis1 = redis.Redis(host="redis", port=6379, db=1, password="MvY4bQ7uN3", decode_responses=True)

# load image embedding vectors
with open("./notebooks/embedding_dict.pkl", "rb") as f:
    embedding_dict = pickle.load(f)

img_embeddings = np.array(list(embedding_dict.values()))

dimension = 1000
index = faiss.IndexFlatL2(dimension)
index.add(img_embeddings)

query_vector = img_embeddings[0]
k = 6  # number of nearest neighbours

# Perform similarity search
D, I = index.search(np.array([query_vector]), k)

# Print the results
print("Indices of nearest neighbors:", I)

# convert I to image_ids
top5_item_idx = I.flatten()[1:]
top5_recommend_item_id = []
key_list = list(embedding_dict.keys())
for idx in top5_item_idx:
    key = key_list[idx]
    top5_recommend_item_id.append(key)

print(top5_recommend_item_id)

redis1.hmset('items_from_faiss', {"top_5": top5_recommend_item_id, "user_id":''})
# redis1.xadd("items_from_faiss", {"top_5": top5_recommend_item_id, "user_id":''}, maxlen=90, approximate=True)