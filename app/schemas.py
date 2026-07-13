from pydantic import BaseModel


# -----------------------
# Sentiment
# -----------------------
class ReviewRequest(BaseModel):
    text: str


class PredictionResponse(BaseModel):
    prediction: str
    confidence: float


# -----------------------
# User Registration
# -----------------------
class UserCreate(BaseModel):
    username: str
    email: str
    password: str


# -----------------------
# Login
# -----------------------
class UserLogin(BaseModel):
    username: str
    password: str


# -----------------------
# JWT Response
# -----------------------
class Token(BaseModel):
    access_token: str
    token_type: str