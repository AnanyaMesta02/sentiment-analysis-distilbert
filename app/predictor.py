import torch
from app.model_loader import tokenizer, model

labels = {
    0: "Negative",
    1: "Positive"
}

def predict_sentiment(text: str):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    )

    with torch.no_grad():
        outputs = model(**inputs)

    prediction = torch.argmax(outputs.logits, dim=1).item()

    confidence = torch.softmax(outputs.logits, dim=1)[0][prediction].item()

    return {
        "prediction": labels[prediction],
        "confidence": round(confidence, 4)
    }