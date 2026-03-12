import pickle

with open("model/email_classifier.pkl", "rb") as f:
    model = pickle.load(f)

examples = [
    "Meu cartão foi cobrado duas vezes",
    "O sistema trava ao fazer login",
    "Quero saber o preço do plano empresarial"
]

predictions = model.predict(examples)

for text, pred in zip(examples, predictions):
    print(f"Texto: {text}")
    print(f"Categoria prevista: {pred}")
    print("-" * 40)