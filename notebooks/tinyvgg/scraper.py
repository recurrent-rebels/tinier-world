#%%
import os
import requests
from pathlib import Path

from bs4 import BeautifulSoup
from urllib.parse import quote

#%%
# Define a function to scrape images with a label from Google Images
def scrape_images_with_label(label, num_images, save_dir):
    # Create the save directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Encode the label for the search query
    query = quote(label)

    # Specify the number of results per page
    results_per_page = 20  # From what I saw the maximum number provided by the URL is 20
    num_pages = (num_images + results_per_page - 1) // results_per_page

    count = 0

    for page in range(num_pages):
        # Calculate the start index for the current page
        start_index = page * results_per_page

        # URL of the search engine (Google Images in this case)
        search_url = f"https://www.google.com/search?q={query}&tbm=isch&num={results_per_page}&start={start_index}"

        # Send an HTTP GET request to the search URL
        response = requests.get(search_url)
        response.raise_for_status()

        # Parse the HTML content using BeautifulSoup
        soup = BeautifulSoup(response.content, 'html.parser')

        # Find all image tags (img) in the HTML
        image_tags = soup.find_all('img')
        print(f"Found {len(image_tags)} images.")
        # Download and save the images
        for img_tag in image_tags:
            image_url = img_tag.get('src')

            # Check if the image URL is valid
            if image_url and image_url.startswith("http"):
                try:
                    # Send an HTTP GET request to download the image
                    img_response = requests.get(image_url, stream=True)
                    img_response.raise_for_status()

                    # Save the image to the save directory
                    img_filename = os.path.join(save_dir, f"{label}_{count}.jpg")
                    with open(img_filename, 'wb') as img_file:
                        for chunk in img_response.iter_content(chunk_size=8192):
                            img_file.write(chunk)

                    count += 1

                    if count >= num_images:
                        break

                except Exception as e:
                    print(f"Error downloading image: {e}")

        if count >= num_images:
            break

    print(f"Downloaded {count} images of {label}.")

#%%
# Specify the directory path
directory_path = Path("../../dataset/train/")

if directory_path.is_dir():
    print(f"{directory_path} directory exists.")
else:
    print(f"{directory_path} directory does not exist.")

#%%
# Download images with specified label
label = "horse"
num_images = 150
save_dir = directory_path / label
scrape_images_with_label(label=label, num_images=num_images, save_dir=save_dir)

# %%
from tokenizers import Tokenizer, trainers, models, pre_tokenizers, decoders

# %%
# Initialize a tokenizer
tokenizer = Tokenizer(models.BPE())

# Initialize a pre-tokenizer
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

# Train the tokenizer model
trainer = trainers.BpeTrainer(vocab_size=2000, min_frequency=2)
files = ["text.txt"]  # specify your file path(s) here
tokenizer.train(files, trainer)

# Initialize a decoder
tokenizer.decoder = decoders.BPEDecoder()

# Test the tokenizer
output = tokenizer.encode("Hello, world!")
print(output.tokens)






# %%
