from flask import Flask, request
import joblib
import os
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

MODEL_PATH = "sentiment_model.pkl"
VEC_PATH = "tfidf_vectorizer.pkl"

if os.path.exists(MODEL_PATH) and os.path.exists(VEC_PATH):
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VEC_PATH)
else:
    data = fetch_20newsgroups(subset='train',
                              categories=['rec.sport.hockey', 'talk.politics.mideast'])
    X, y = data.data, data.target
    vectorizer = TfidfVectorizer(max_features=5000)
    X_vec = vectorizer.fit_transform(X)
    model = LogisticRegression(max_iter=200)
    model.fit(X_vec, y)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VEC_PATH)

@app.route("/", methods=["GET", "POST"])
def index():
    result = ""
    if request.method == "POST":
        text = request.form.get("text", "")
        vec = vectorizer.transform([text])
        pred = model.predict(vec)[0]
        label = "Olumlu" if pred == 1 else "Olumsuz"
        result = f"<h2>Sonuç: {label}</h2>"
    return f"""
    <html>
    <head><title>AI Duygu Analizi</title></head>
    <body style="text-align:center; font-family:sans-serif; margin-top:50px;">
      <h1>Yapay Zeka ile Duygu Analizi</h1>
      <form method="POST">
        <textarea name="text" rows="6" cols="60" placeholder="Bir metin yaz..."></textarea><br><br>
        <input type="submit" value="Tahmin Et">
      </form>
      {result}
    </body>
    </html>
    """

if __name__ == "__main__":
    app.run(debug=True)