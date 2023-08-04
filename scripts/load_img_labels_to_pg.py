#%%
import os
import pandas
import dotenv
import psycopg2
from pathlib import Path
#%%
dotenv.load_dotenv("/root/tinier-world/.env")
#%%
# rootdir = "/root/tinier-world/dataset/train"
data_path = Path("../dataset/")
images = []
labels = []

if data_path.is_dir():
    print(f"{data_path} directory exists.")
else:
    print(f"{data_path} directory does not exist.")
#%%
for subdir, dirs, files in os.walk(data_path):
  for file in files:
    # Only include .jpg files
    if file.endswith(".jpg"):
      # Extract the label from the subdir and file name from file
      label = os.path.basename(subdir)
      image_id = file.split(".")[0] # .jpg
      images.append(image_id)
      labels.append(label)

print(f"There are {len(images)} images and {len(labels)} labels.")
#%%
# Create a Pandas DataFrame of the image IDs and labels
csv_path = data_path / "output.csv"

df = pandas.DataFrame({"id": images, "label": labels})
df.to_csv("../dataset/output.csv", index=False)
#%%
# Plot label distribution on bar chart
label_counts = df["label"].value_counts()
ax = label_counts.plot(kind="bar", figsize=(10, 6))
ax.set_title("Label Distribution")
ax.set_xlabel("Labels")
ax.set_ylabel("Count")
# %%
# Connect to PostgreSQL
USER = "root"
DATABASE = "W9sV6cL2dX"
PASSWORD = "E5rG7tY3fH"
c = psycopg2.connect(host="localhost", user=USER, port=5432, database=DATABASE, password=PASSWORD)
c.autocommit = True
cursor = c.cursor()
# %%
# Create img_labels table
cursor.execute("""
  BEGIN;

  CREATE TABLE IF NOT EXISTS img_labels (
    item_key   UUID        PRIMARY KEY,
    label      VARCHAR(20) NOT NULL
  );

  COMMIT;
""")
# %%
#  Insert data into img_labels table
for index, row in df.iterrows():
  cursor.execute("""
    INSERT INTO img_labels (item_key, label)
    VALUES (%s, %s);
  """, (row["id"], row["label"]))
# %%
# Check that data was inserted by selecting the first 5 rows
cursor.execute("SELECT * FROM img_labels LIMIT 5;")
labels = cursor.fetchmany(2)
print("labels", labels)
# %%
# Check that the count of rows in the table matches the number of images
cursor.execute("SELECT COUNT(*) FROM img_labels;")
count = cursor.fetchone()[0]
print("Count:", count)
# %%
