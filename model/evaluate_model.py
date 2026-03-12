import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# carregar dataset
df = pd.read_csv("data/emails.csv")

X = df["texto"]
y = df["categoria"]

labels = ["financeiro", "tecnico", "comercial"]

# separar treino e teste mantendo proporção das classes
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

# carregar modelo treinado
with open("model/email_classifier.pkl", "rb") as f:
    model = pickle.load(f)

# prever
y_pred = model.predict(X_test)

print("Classification Report:\n")
print(classification_report(y_test, y_pred, labels=labels))

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred, labels=labels))