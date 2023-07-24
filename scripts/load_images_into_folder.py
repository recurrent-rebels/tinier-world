import requests
from PIL import Image
from io import BytesIO

url = 'http://135.181.118.171:7070/items/156600' # gives 54 items, 20 images

response = requests.get(url)
# Raise an exception if the GET request was unsuccessful
response.raise_for_status()

items = response.json()
for item in items:
    if item['type'] == 'img':
        image_id = item['item_key']
        # transform the image to https://tiny-images-jk9apq.s3.us-east-1.amazonaws.com/000ccbdf-3362-45a5-b1d0-2410f112cd90.jpg
        response = requests.get(f"https://tiny-images-jk9apq.s3.us-east-1.amazonaws.com/{image_id}.jpg")
        img = Image.open(BytesIO(response.content))
        img.save(f"../images/{image_id}.jpg")
        




