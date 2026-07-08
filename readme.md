# 🚀 Sentiment Analysis & Review Intelligence

A production-ready **Sentiment Analysis API** built using **DistilBERT**, **FastAPI**, and **Docker**. The model is fine-tuned on the IMDb movie reviews dataset to classify reviews as **Positive** or **Negative**.

---

## 📌 Project Overview

This project demonstrates an end-to-end Natural Language Processing (NLP) workflow:

- Fine-tuning a pre-trained DistilBERT model
- Building a REST API using FastAPI
- Containerizing the application with Docker
- Testing through Swagger UI
- Preparing the project for cloud deployment

---

## ✨ Features

- Binary Sentiment Classification
- Fine-tuned DistilBERT Model
- REST API with FastAPI
- Interactive Swagger Documentation
- Docker Support
- Clean Project Structure
- Ready for Cloud Deployment

---

## 🛠 Tech Stack

### Programming Language
- Python

### Machine Learning
- Hugging Face Transformers
- DistilBERT
- PyTorch

### Backend
- FastAPI
- Uvicorn

### Tools
- Docker
- Git
- GitHub
- VS Code

### Dataset
- IMDb Movie Reviews Dataset

---

## 📂 Project Structure

```
sentiment-analysis-distilbert/
│
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── predictor.py
│   ├── model_loader.py
│   └── schemas.py
│
├── model/
│
├── train.py
├── train_model.py
├── requirements.txt
├── Dockerfile
├── .gitignore
├── README.md
└── test_model.py
```

---

## ⚙ Installation

### Clone Repository

```bash
git clone https://github.com/AnanyaMesta02/sentiment-analysis-distilbert.git

cd sentiment-analysis-distilbert
```

---

### Create Virtual Environment

```bash
python -m venv venv
```

Windows

```bash
venv\Scripts\activate
```

Linux / Mac

```bash
source venv/bin/activate
```

---

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🎯 Train the Model

```bash
python train_model.py
```

The trained model will be saved inside the **model/** folder.

---

## ▶ Run the API

```bash
python -m uvicorn app.main:app --reload
```

Open

```
http://127.0.0.1:8000/docs
```

Swagger UI will appear.

---

## 📌 API Endpoint

### POST `/predict`

Request

```json
{
  "text": "This movie was absolutely amazing!"
}
```

Response

```json
{
  "prediction": "Positive",
  "confidence": 0.9987
}
```

---

## 🐳 Docker

### Build Image

```bash
docker build -t sentiment-api .
```

### Run Container

```bash
docker run -p 8000:8000 sentiment-api
```

Open

```
http://localhost:8000/docs
```

---

## 📊 Model Information

Model

- DistilBERT

Task

- Binary Sentiment Classification

Dataset

- IMDb Movie Reviews

Framework

- Hugging Face Transformers

Backend

- PyTorch

---

## 📷 Demo

### Swagger UI

<img width="900" alt="Swagger UI" src="images/SWAGGER UI.png">

### Prediction Example

<img width="900" alt="Prediction" src="images/PREDICTION RESPONSE.png">

*(Replace these with your own screenshots after creating an `images/` folder.)*

---

## 🚀 Future Improvements

- Deploy on AWS Elastic Beanstalk
- Deploy on Hugging Face Spaces
- Upload model to Hugging Face Hub
- Batch Prediction API
- Authentication using JWT
- Logging and Monitoring
- CI/CD using GitHub Actions

---

## 👩‍💻 Author

**Ananya Mesta**

GitHub

https://github.com/AnanyaMesta02

LinkedIn

https://www.linkedin.com/in/ananyamesta/

---

## ⭐ If you found this project useful, consider giving it a star!
