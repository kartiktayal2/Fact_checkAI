from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

# ----------------------------
# Load and train model ONCE
# ----------------------------
data = pd.read_csv("data/claims.csv")

def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text

data["clean_text"] = data["text"].apply(clean_text)

vectorizer = TfidfVectorizer(max_features=3000)
X = vectorizer.fit_transform(data["clean_text"])
y = data["label"]

model = LogisticRegression()
model.fit(X, y)

# Global variables to store result
last_result = None
last_confidence = None

def check_claim(claim, threshold=0.6):
    cleaned = clean_text(claim)
    vector = vectorizer.transform([cleaned])
    probs = model.predict_proba(vector)[0]
    confidence = max(probs)
    prediction = probs.argmax()

    if confidence < threshold:
        return "Not Enough Evidence", confidence
    return ("Likely TRUE" if prediction == 1 else "Likely FALSE"), confidence

# ----------------------------
# Routes
# ----------------------------
@app.route("/", methods=["GET", "POST"])
def home():
    global last_result, last_confidence

    if request.method == "POST":
        claim = request.form["claim"]
        last_result, last_confidence = check_claim(claim)
        return redirect(url_for("home"))  # 🔥 KEY FIX

    return render_template(
        "index.html",
        result=last_result,
        confidence=last_confidence
    )

if __name__ == "__main__":
    app.run(debug=True)
