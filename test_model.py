from transformers import pipeline

# Load a pre-trained sentiment analysis pipeline
classifier = pipeline("sentiment-analysis")

# Test sentence
text = "I love learning Machine Learning with Hugging Face."

# Predict sentiment
result = classifier(text)

print(result)