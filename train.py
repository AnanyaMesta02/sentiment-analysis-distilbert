from datasets import load_dataset
from transformers import AutoTokenizer

# Load IMDb dataset
dataset = load_dataset("imdb")

# Load DistilBERT tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# Tokenization function
def tokenize_function(example):
    return tokenizer(
        example["text"],
        padding="max_length",
        truncation=True,
        max_length=256
    )

print("Tokenizing dataset...")

# Tokenize entire dataset
tokenized_dataset = dataset.map(tokenize_function, batched=True)

print("Tokenization Completed!")

print(tokenized_dataset["train"][0])