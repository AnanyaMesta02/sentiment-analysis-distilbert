from fastapi import FastAPI, HTTPException
from app.schemas import ReviewRequest, PredictionResponse
from app.predictor import predict_sentiment
from fastapi import FastAPI, HTTPException
from app.database import engine
from app.database import Base
from app.schemas import UserCreate
from app.database import SessionLocal
from app.models import User
from app.auth import hash_password
from sqlalchemy import or_
from app.schemas import UserLogin
from app.schemas import Token
from fastapi import Depends
from app.auth import verify_token
from app.auth import verify_password
from app.auth import create_access_token
from app.database import engine
from app.models import Base

Base.metadata.create_all(bind=engine)
app = FastAPI(
    title="Sentiment Analysis API",
    description="Sentiment Analysis using DistilBERT and FastAPI",
    version="1.0.0"
)
Base.metadata.create_all(bind=engine)

@app.get("/")
def home():
    return {
        "message": "Welcome to Sentiment Analysis API"
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(
    review: ReviewRequest,
    username: str = Depends(verify_token)
):
    try:
        return predict_sentiment(review.text)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
    try:
        return predict_sentiment(review.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@app.post("/register")
def register(user: UserCreate):
    try:
        db = SessionLocal()

        existing_user = db.query(User).filter(
            User.username == user.username
        ).first()

        if existing_user:
            db.close()
            raise HTTPException(
                status_code=400,
                detail="Username already exists"
            )

        new_user = User(
            username=user.username,
            email=user.email,
            hashed_password=hash_password(user.password)
        )

        db.add(new_user)
        db.commit()
        db.refresh(new_user)
        db.close()

        return {
            "message": "User registered successfully"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/login", response_model=Token)
def login(user: UserLogin):

    db = SessionLocal()

    db_user = db.query(User).filter(
        User.username == user.username
    ).first()

    if not db_user:
        db.close()
        raise HTTPException(
            status_code=401,
            detail="Invalid username or password"
        )

    if not verify_password(
        user.password,
        db_user.hashed_password
    ):
        db.close()
        raise HTTPException(
            status_code=401,
            detail="Invalid username or password"
        )

    token = create_access_token(
        {"sub": db_user.username}
    )

    db.close()

    return {
        "access_token": token,
        "token_type": "bearer"
    }