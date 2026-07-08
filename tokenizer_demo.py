from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

review = "I absolutely love this movie!"

encoded = tokenizer(review)

print("Input IDs:")
print(encoded["input_ids"])

print("\nTokens:")
print(tokenizer.convert_ids_to_tokens(encoded["input_ids"]))