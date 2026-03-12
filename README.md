# Customer Support AI

AI-powered system for automatic classification and prioritization of customer support emails using Machine Learning and FastAPI.

This project demonstrates how Natural Language Processing (NLP) can be used to automate support ticket triage by categorizing incoming emails and assigning priority levels.

## Features

- Email classification into support categories
- Automatic ticket prioritization
- Machine Learning text processing pipeline
- REST API built with FastAPI
- Interactive API documentation (Swagger UI)

## Categories

The model classifies incoming emails into:

- Technical
- Financial
- Commercial

## Priority Detection

The system automatically assigns priority levels based on the email content:

- **High** → urgent issues such as billing errors or system access problems  
- **Medium** → standard technical or financial inquiries  
- **Low** → commercial or informational requests

## Machine Learning Pipeline

The classification model is built using:

- TF-IDF vectorization
- Logistic Regression
- Scikit-learn pipeline

Dataset example:
texto,categoria
"Meu pagamento foi cobrado duas vezes",financeiro
"O sistema está travando ao fazer login",tecnico
"Gostaria de saber o preço do plano premium",comercial

## API

The system exposes a REST API built with FastAPI.

### Start the API

Run:
uvicorn api.app:app --reload

Then open:
http://127.0.0.1:8000/docs

to access the interactive API documentation.

### Example Request

POST `/classify`

Input:
Meu pagamento foi cobrado duas vezes

Response:

```json
{
  "email": "Meu pagamento foi cobrado duas vezes",
  "categoria": "financeiro",
  "prioridade": "alta"
}
```
## Project Structure

customer-support-ai
│
├── api
│   └── app.py
│
├── data
│   └── emails.csv
│
├── model
│   ├── train_model.py
│   └── email_classifier.pkl
│
├── utils
│
└── README.md

## Technologies

Python

FastAPI

Scikit-learn

Pandas

Uvicorn

## Future Improvements

Possible enhancements for the project:

Email summarization using NLP

Model performance evaluation

Larger training dataset

Deployment with Docker

Integration with real support systems

## Author

Lucas Cunha de Alvarenga