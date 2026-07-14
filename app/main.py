from fastapi import FastAPI
from app.schemas import ReviewRequest, PredictionResponse
from app.predictor import predict_sentiment
from fastapi import FastAPI, HTTPException

app = FastAPI(
    title="Sentiment Analysis API",
    description="Sentiment Analysis using DistilBERT and FastAPI",
    version="1.0.0"
)


@app.get("/")
def home():
    return {
        "message": "Welcome to Sentiment Analysis API"
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(review: ReviewRequest):
    try:
        return predict_sentiment(review.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))