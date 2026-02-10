import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Load dataset (with clean_text column already created)
data = pd.read_csv("data/claims.csv")

# Recreate preprocessing (simple + consistent)
def clean_text(text):
    import string
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text

data["clean_text"] = data["text"].apply(clean_text)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=3000)
X = vectorizer.fit_transform(data["clean_text"])
y = data["label"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred, zero_division=0))

# ----------------------------
# Confidence-based prediction
# ----------------------------
def check_claim(claim, threshold=0.6):
    cleaned = clean_text(claim)
    vector = vectorizer.transform([cleaned])

    probs = model.predict_proba(vector)[0]
    confidence = max(probs)
    prediction = probs.argmax()

    if confidence < threshold:
        return f"NOT ENOUGH EVIDENCE (confidence={confidence:.2f})"

    if prediction == 1:
        return f"Likely TRUE (confidence={confidence:.2f})"
    else:
        return f"Likely FALSE (confidence={confidence:.2f})"
    


