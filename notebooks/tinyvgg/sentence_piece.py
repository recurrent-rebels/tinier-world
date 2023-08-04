#%%
from transformers import XLMRobertaTokenizer, XLMRobertaModel
import sentencepiece as spm
import torch

# A human vocabulary can be though of as words, spaces and punctuation. 
# Humans understand the meaning of text through words, spaces and punctuation.

# A tokenizer's (such as SentencePiece) vocabulary can be thought of as tokens.
# The tokenizer decides what tokens it has in its vocabulary (done when training the model).

# Each token has an associated ID.
vocab = {
    "<unk>": 0,
    "<s>": 1,
    "</s>": 2,
    "the": 3,
    "a": 4,
    "in": 5,
    "of": 6,
    "and": 7,
    "to": 8,
    "cat": 9,
    "c": 10,
    "at": 11
    # ... and so on for all the tokens
}

#%%
model_name = 'xlm-roberta-base'

# The model
model = XLMRobertaModel.from_pretrained(model_name)

# The corresponding tokenizer
tokenizer = XLMRobertaTokenizer.from_pretrained(model_name)
print(tokenizer)

# Step 1: Use the tokenizer to tokenize the text
# Step 2: Use the model to get embeddings for each token

#%%
# Tokenize some text
text = "Hello everyone!"
input_ids = tokenizer.encode(text, return_tensors="pt")
print(f"English input_ids: {input_ids}")

#%%
# Map the IDs in the vocabulary to their respective tokens.
input_ids_1d = input_ids.squeeze(dim=0)
tokens = tokenizer.convert_ids_to_tokens(input_ids_1d)
print(f"Tokens: {tokens}")

#%%
# Tokenize some Mandarin
text_mandarin = "大家好"
input_mandarin_ids = tokenizer.encode(text_mandarin, return_tensors="pt")
print(f"Mandarin input_ids: {input_mandarin_ids}")

#%%
# Map the IDs in the vocabulary to their respective tokens.
input_ids_mandarin_1d = input_mandarin_ids.squeeze(dim=0)
tokens_mandarin = tokenizer.convert_ids_to_tokens(input_ids_mandarin_1d)
print(f"Tokens Mandarin: {tokens_mandarin}")

#%%
# Run the model to get embeddings
print(input_mandarin_ids.shape) # torch.Size([1, 5])
outputs_mandarin = model(input_mandarin_ids)
print(outputs_mandarin[0].shape) # torch.Size([1, 5, 768])

outputs = model(input_ids)

#%%
# The last hidden-state is the first element of 'outputs_mandarin', 
# it is a tensor containing the embeddings for each token in the input text.
embeddings = outputs.last_hidden_state
embeddings_mandarin = outputs_mandarin.last_hidden_state
print(embeddings_mandarin.shape)

#%%
# Find the mean embedding accross all the token embeddings
sentence_embedding = torch.mean(embeddings, dim=1)
sentence_embedding_mandarin = torch.mean(embeddings_mandarin, dim=1)
print(sentence_embedding_mandarin.shape) # torch.Size([1, 768])

# %%
from torch.nn.functional import cosine_similarity

# Calculate the cosine similarity between the two sentence embeddings
similarity = cosine_similarity(sentence_embedding, sentence_embedding_mandarin)
print("Similarity Between Sentence Embeddings: ", similarity.item())

#%%
# Tokenize some seemingly different text
text_cheese = "You are a disgusting pie eating world hating frog!"
input_ids_mean = tokenizer.encode(text_cheese, return_tensors="pt")
outputs_cheese = model(input_ids_mean)
embeddings_cheese = outputs_cheese.last_hidden_state
sentence_embedding_cheese = torch.mean(embeddings_cheese, dim=1)

# Calculate the cosine similarity between the two sentence embeddings
similarity_cheese = cosine_similarity(sentence_embedding, sentence_embedding_cheese)
print("Similarity Between Sentence Embeddings: ", similarity_cheese.item())