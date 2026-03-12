from fastapi import FastAPI
import pickle

app = FastAPI()

with open("model/email_classifier.pkl", "rb") as f:
    model = pickle.load(f)

def definir_prioridade(categoria: str, text: str) -> str:
    text_lower = text.lower()

    if categoria == "financeiro":
        if "cobrado" in text_lower or "duplicado" in text_lower or "pagamento" in text_lower:
            return "alta"
        return "media"

    if categoria == "tecnico":
        if "não consigo" in text_lower or "nao consigo" in text_lower or "travando" in text_lower:
            return "alta"
        return "media"

    if categoria == "comercial":
        return "baixa"

    return "media"

@app.get("/")
def home():
    return {"message": "Customer Support AI is running"}

@app.post("/classify")
def classify_email(text: str):
    prediction = model.predict([text])[0]
    prioridade = definir_prioridade(prediction, text)

    return {
        "email": text,
        "categoria": prediction,
        "prioridade": prioridade
    }