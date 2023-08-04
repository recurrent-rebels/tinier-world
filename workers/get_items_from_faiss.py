import faiss
import pickle
import numpy as np
import redis
import psycopg2
import fastapi
import numpy as np

app = fastapi.FastAPI()
# user id # recommended_items_id 
# /home -user_id-> this server (find longest seen item in db for user, then use this item_id to get embedding)
#        <- items -
# get time_spent_on_item

connection = psycopg2.connect(host="postgres", user="root", port=5432, database="W9sV6cL2dX", password="E5rG7tY3fH")
cursor = connection.cursor()

dimension = 1000

embeddings_dict = {}
img_embeddings = []

def load_embeddings_into_memory():
    get_embeddings_query = f"SELECT item_id, embedding FROM item_embeddings"
    global embeddings_dict
    global img_embeddings
    cursor.execute(get_embeddings_query)
    embeddings_dict = dict(cursor.fetchall())
    print('len(embeddings_dict.keys()', len(embeddings_dict.keys()))
    img_embeddings = np.array(list(embeddings_dict.values()), dtype=np.float32)
    print('type(img_embeddings)', img_embeddings.shape)


def get_select_query(user_id):
    """
    Generate a SQL query to fetch the item_id with maximum time spent for a specific user.

    Parameters:
    user_id (int): The ID of the user.

    Returns:
    str: A SQL query.
    """
    # Parameterized query to prevent SQL injection
    select_query = f"""
        SELECT item_id
        FROM time_spent_on_item
        WHERE user_id = '{user_id}' AND time_spent = (
            SELECT MAX(time_spent) 
            FROM time_spent_on_item 
            WHERE user_id = '{user_id}'
        )
        LIMIT 1;
    """
    return select_query


@app.get("/")
def read_root(request: fastapi.Request):
    user_id = request.headers.get("user")
    select_query = get_select_query(user_id)
    cursor.execute(select_query)
    longest_seen_item_id = cursor.fetchall()
    print('user', user_id)
    # print("select_query", select_query)
    # print("longest_seen_item_id", longest_seen_item_id)
    if (len(longest_seen_item_id) == 0):
        return None
    longest_seen_item_id = longest_seen_item_id[0][0] # [('945a23c6-51be-4359-b717-4d9c8b71d4e2',)] -> '945a23c6-51be-4359-b717-4d9c8b71d4e2'
    global img_embeddings
    # print('img_embeddings.shape', img_embeddings.shape)
    index = faiss.IndexFlatL2(dimension)
    index.add(img_embeddings)
    k = 6  # number of nearest neighbours
    query_vector = embeddings_dict.get(longest_seen_item_id)
    if (query_vector == None): # if item is new and we haven't saved it to the db yet 
        return None
    # Perform similarity search
    D, I = index.search(np.array([query_vector], dtype=np.float32), k+1)

    top5_item_idx = I.flatten()[1:]
    top5_recommend_item_id = []
    key_list = list(embeddings_dict.keys())
    for idx in top5_item_idx:
        key = key_list[idx]
        top5_recommend_item_id.append(key)

    return top5_recommend_item_id


load_embeddings_into_memory()