import requests
from PIL import Image
from io import BytesIO
import psycopg2
import os

# url = 'http://135.181.118.171:7070/items/0' # gives 54 items, 20 images
connection = psycopg2.connect(host="localhost", user="root", port=5432, database="W9sV6cL2dX", password="E5rG7tY3fH")
cursor = connection.cursor()

select_query = "SELECT item_key, bucket_key FROM items WHERE type = 'img'"

# Raise an exception if the GET request was unsuccessful
cursor.execute(select_query)
for item in cursor:
    image_id = item[0]
    bucket_key = item[1]

    image_path = os.path.join('../images', f"{image_id}.jpg")
    if os.path.isfile(image_path):
        continue
    # transform the image to https://tiny-images-jk9apq.s3.us-east-1.amazonaws.com/000ccbdf-3362-45a5-b1d0-2410f112cd90.jpg
    url = f"https://{bucket_key}.s3.us-east-1.amazonaws.com/{image_id}.jpg"
    response = requests.get(url)
    try:
        img = Image.open(BytesIO(response.content))
        img.save(f"../images/{image_id}.jpg")
    except:
        print(f'url {url} cannot save')




