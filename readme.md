# 🚀 Sentiment Analysis & Review Intelligence

A production-ready REST API for sentiment analysis built using **DistilBERT, FastAPI, Docker, Hugging Face Transformers**, and **deployed on Railway**.

The model is fine-tuned on the **IMDb Movie Reviews** dataset to classify reviews as **Positive** or **Negative**.

---

## 🌐 Live Demo

**Railway Deployment**

https://sentiment-analysis-distilbert-production.up.railway.app

---

## 📄 API Documentation

Interactive Swagger UI

https://sentiment-analysis-distilbert-production.up.railway.app/docs

---

## 🤗 Hugging Face Model

https://huggingface.co/Ananya02/sentiment-analysis-distilbert

The deployed API downloads the trained DistilBERT model directly from the Hugging Face Hub during startup, keeping the GitHub repository lightweight while ensuring the latest model is always available.

---

# 📌 Project Overview

This project demonstrates an end-to-end Natural Language Processing (NLP) workflow:

- Fine-tuning a pre-trained DistilBERT model
- Building a REST API using FastAPI
- Containerizing the application using Docker
- Hosting the trained model on Hugging Face Hub
- Deploying the API publicly using Railway
- Testing through Swagger UI

---

# ✨ Features

- Binary Sentiment Classification
- Fine-tuned DistilBERT Model
- FastAPI REST API
- Interactive Swagger Documentation
- Dockerized Application
- Hugging Face Model Hosting
- Railway Cloud Deployment
- Clean Project Structure

---

# 🛠 Tech Stack

## Programming Language

- Python

## Machine Learning

- DistilBERT
- Hugging Face Transformers
- PyTorch

## Backend

- FastAPI
- Uvicorn

## Deployment

- Docker
- Railway
- Hugging Face Hub

## Version Control

- Git
- GitHub

## Development

- VS Code

## Dataset

- IMDb Movie Reviews Dataset

---

# 🏗 System Architecture

```text
                User
                  │
                  ▼
         Railway Deployment
                  │
                  ▼
             FastAPI API
                  │
                  ▼
     DistilBERT Sentiment Model
                  │
                  ▼
      Positive / Negative Prediction
```

---

# 📂 Project Structure

```text
sentiment-analysis-distilbert/
│
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── model_loader.py
│   ├── predictor.py
│   └── schemas.py
│
├── model/
│
├── images/
│   ├── SWAGGER UI.png
│   └── PREDICTION RESPONSE.png
│
├── requirements.txt
├── Dockerfile
├── .gitignore
├── README.md
├── train_model.py
├── test_model.py
└── tokenizer_demo.py
```

---

# ⚙ Installation

## Clone Repository

```bash
git clone https://github.com/AnanyaMesta02/sentiment-analysis-distilbert.git

cd sentiment-analysis-distilbert
```

---

## Create Virtual Environment

Windows

```bash
python -m venv venv
venv\Scripts\activate
```

Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 🎯 Train the Model

```bash
python train_model.py
```

The trained model will be saved in the **model/** folder.

---

# ▶ Run the API

```bash
python -m uvicorn app.main:app --reload
```

Open:

```
http://127.0.0.1:8000/docs
```

---

# 📌 API Endpoint

## POST `/predict`

### Request

```json
{
    "text": "This movie was absolutely amazing!"
}
```

### Response

```json
{
    "prediction": "Positive",
    "confidence": 0.9006
}
```

---

# 🐳 Docker

## Build Docker Image

```bash
docker build -t sentiment-api .
```

## Run Docker Container

```bash
docker run -p 8000:8000 sentiment-api
```

Open:

```
http://localhost:8000/docs
```

---

# 📊 Model Information

| Property | Value |
|----------|-------|
| Model | DistilBERT |
| Task | Binary Sentiment Classification |
| Dataset | IMDb Movie Reviews |
| Framework | Hugging Face Transformers |
| Backend | PyTorch |

---

# 📷 Demo

## Swagger UI

![Swagger UI](images/SWAGGER%20UI.png)

---

## Prediction Example

![Prediction Response](images/PREDICTION%20RESPONSE.png)

---

# 🚀 Deployment Status

- ✅ Status: Live
- 🌐 Platform: Railway
- 🤗 Model Hosting: Hugging Face Hub
- 🐳 Dockerized
- 📄 Swagger Documentation Available

---

# 🔮 Future Improvements

- Multi-class sentiment classification
- Batch prediction API
- JWT Authentication
- Request logging
- Monitoring and analytics
- CI/CD with GitHub Actions
- Model versioning

---

# 👩‍💻 Author

**Ananya Mesta**

### GitHub

https://github.com/AnanyaMesta02

### LinkedIn

https://www.linkedin.com/in/ananyamesta/

---

⭐ **If you found this project useful, consider giving it a star!**