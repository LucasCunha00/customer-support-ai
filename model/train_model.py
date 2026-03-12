import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Carrega o dataset
df = pd.read_csv("data/emails.csv")

# Entradas e saídas
X = df["texto"]
y = df["categoria"]

# Pipeline de vetorização + classificação
model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", LogisticRegression())
])

# Treinamento
model.fit(X, y)

# Salva o modelo treinado
with open("model/email_classifier.pkl", "wb") as f:
    pickle.dump(model, f)

print("Modelo treinado com sucesso e salvo em model/email_classifier.pkl")