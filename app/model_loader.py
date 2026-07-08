from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Get absolute path to the project root
BASE_DIR = Path(__file__).resolve().parent.parent

# Absolute path to model folder
MODEL_PATH = BASE_DIR / "model"

print("Loading model from:", MODEL_PATH)

tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH))

model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_PATH))

print("Model loaded successfully!")