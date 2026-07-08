from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
import evaluate
import numpy as np

# ==========================================
# Load IMDb Dataset
# ==========================================

print("Loading IMDb Dataset...")

dataset = load_dataset("imdb")

# ==========================================
# Load Tokenizer
# ==========================================

print("Loading Tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    "distilbert-base-uncased"
)

# ==========================================
# Tokenize Dataset
# ==========================================

def tokenize_function(example):
    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=256,
    )

print("Tokenizing Dataset...")

tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
)

# ==========================================
# Small Dataset (Fast Training)
# ==========================================

train_dataset = (
    tokenized_dataset["train"]
    .shuffle(seed=42)
    .select(range(2000))
)

test_dataset = (
    tokenized_dataset["test"]
    .shuffle(seed=42)
    .select(range(500))
)

# ==========================================
# Load Model
# ==========================================

print("Loading DistilBERT Model...")

model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2,
)

# ==========================================
# Accuracy Metric
# ==========================================

accuracy = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    return accuracy.compute(
        predictions=predictions,
        references=labels,
    )


# ==========================================
# Training Arguments
# ==========================================

training_args = TrainingArguments(
    output_dir="./results",

    eval_strategy="epoch",

    save_strategy="epoch",

    learning_rate=2e-5,

    per_device_train_batch_size=16,

    per_device_eval_batch_size=16,

    num_train_epochs=2,

    weight_decay=0.01,

    logging_dir="./logs",

    logging_steps=100,

    load_best_model_at_end=True,

    metric_for_best_model="accuracy",

    report_to="none",
)

# ==========================================
# Trainer
# ==========================================

trainer = Trainer(
    model=model,

    args=training_args,

    train_dataset=train_dataset,

    eval_dataset=test_dataset,

    tokenizer=tokenizer,

    compute_metrics=compute_metrics,
)

# ==========================================
# Train
# ==========================================

print("\nTraining Started...\n")

trainer.train()

print("\nTraining Completed!\n")

# ==========================================
# Evaluate
# ==========================================

results = trainer.evaluate()

print("\nEvaluation Results")

print(results)

# ==========================================
# Save Model
# ==========================================

trainer.save_model("./model")

tokenizer.save_pretrained("./model")

print("\nModel Saved Successfully!")

print("Location : ./model")