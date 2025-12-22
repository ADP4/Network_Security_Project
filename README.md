# Network Security – Phishing Detection System

An end-to-end machine learning–powered phishing detection system built using structured URL and webpage features.
The project covers the complete ML lifecycle — data ingestion, validation, transformation, model training, evaluation, experiment tracking, containerization, and cloud-ready deployment setup.

## Project Overview

Phishing attacks remain a major threat in network security. This project focuses on detecting phishing websites using supervised machine learning models trained on engineered security-related features such as URL structure, SSL state, domain age, redirection behavior, and HTML signals.

The system is designed as a production-grade ML pipeline, not just a notebook experiment.

## Key Features

-End-to-end ML pipeline with modular components
-Automated data ingestion, validation, and transformation
-Multiple classification models with hyperparameter tuning
-Model selection using F1-score and overfitting checks
-Experiment tracking using MLflow
-REST API for training and prediction using FastAPI
-Dockerized application with CI/CD pipeline for ECR
-Cloud-ready architecture (AWS EC2 / ECS compatible)

## Architecture
Data Ingestion
      ↓
Data Validation
      ↓
Data Transformation
      ↓
Model Training & Selection
      ↓
Model Registry (MLflow)
      ↓
FastAPI Inference Service
      ↓
Docker + AWS ECR

## Models Used

The following models were evaluated and compared:

-Logistic Regression
-Random Forest
-Gradient Boosting
-AdaBoost
-Support Vector Machine (RBF Kernel)
-XGBoost

Model selection was performed using test F1-score, with additional checks for overfitting and underfitting.

## Evaluation Metrics

-F1-Score (primary metric)
-Precision
-Recall

A full comparison report is generated for all candidate models during training.

## Tech Stack

Language: Python
-ML: scikit-learn, XGBoost
-API: FastAPI
-Experiment Tracking: MLflow
-Containerization: Docker
-CI/CD: GitHub Actions
-Cloud: AWS (ECR, EC2 compatible)
-Database: MongoDB (for ingestion)

### How to Run Locally
1. Clone the repository
git clone https://github.com/ADP4/Network_Security_Project.git
cd Network_Security_Project

2. Create virtual environment & install dependencies
pip install -r requirements.txt

3. Train the model
uvicorn app:app --host 0.0.0.0 --port 8000

Then open:
http://localhost:8000/train

4. Make predictions
Upload a CSV file using:

http://localhost:8000/predict

## Docker Support

Build the Docker image:
docker build -t network-security .

Run the container:
docker run -p 8000:8000 network-security

## CI/CD Pipeline
Automated build using GitHub Actions

Docker image pushed to AWS ECR

Ready for deployment on EC2 / ECS

Note: Final EC2 deployment was prepared but not executed due to account constraints. The architecture and CI/CD pipeline are fully deployment-ready.

## Project Structure
networksecurity/
│
├── components/
│   ├── data_ingestion.py
│   ├── data_validation.py
│   ├── data_transformation.py
│   └── model_trainer.py
│
├── pipeline/
│   └── training_pipeline.py
│
├── utils/
│   ├── ml_utils/
│   └── main_utils/
│
├── app.py
├── Dockerfile
├── requirements.txt
└── README.md



**Author**

Anuja D. Parab
Data Scientist | Machine Learning 
📍 Mumbai, India