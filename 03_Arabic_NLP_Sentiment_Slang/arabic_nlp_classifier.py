import re
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

DATA = [
    ("الفيلم جميل جدا وعجبني", "positive"),
    ("الخدمة ممتازة والمكان حلو", "positive"),
    ("التطبيق سريع وسهل الاستخدام", "positive"),
    ("أنا مبسوط جدا من النتيجة", "positive"),
    ("الشغل ده جامد جدا", "positive"),
    ("الموضوع وحش ومش عاجبني", "negative"),
    ("الخدمة سيئة جدا", "negative"),
    ("التطبيق بطيء ومليان مشاكل", "negative"),
    ("أنا زعلان من التجربة", "negative"),
    ("الكلام ده ملوش لازمة", "negative"),
    ("الجو عادي ومفيش حاجة مميزة", "neutral"),
    ("التجربة متوسطة", "neutral"),
    ("مش سيء ومش ممتاز", "neutral"),
    ("المنتج مقبول", "neutral"),
    ("الموضوع عادي", "neutral"),
]

def clean_arabic_text(text):
    text = str(text)
    text = re.sub(r"[إأآا]", "ا", text)
    text = re.sub(r"ى", "ي", text)
    text = re.sub(r"ؤ", "و", text)
    text = re.sub(r"ئ", "ي", text)
    text = re.sub(r"ة", "ه", text)
    text = re.sub(r"[^\u0600-\u06FF\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def train_model():
    df = pd.DataFrame(DATA, columns=["text", "label"])
    df["clean_text"] = df["text"].apply(clean_arabic_text)

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"],
        df["label"],
        test_size=0.25,
        random_state=42,
        stratify=df["label"]
    )

    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
        ("classifier", LogisticRegression(max_iter=1000))
    ])

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, predictions))
    print(classification_report(y_test, predictions))

    joblib.dump(model, "arabic_sentiment_model.pkl")
    print("Saved model as arabic_sentiment_model.pkl")
    return model

def predict_text(model, text):
    cleaned = clean_arabic_text(text)
    prediction = model.predict([cleaned])[0]
    probabilities = model.predict_proba([cleaned])[0]
    labels = model.classes_
    confidence = dict(zip(labels, probabilities))
    return prediction, confidence

if __name__ == "__main__":
    model = train_model()

    print("\nType Arabic text to classify. Type q to quit.")
    while True:
        user_text = input("Enter text: ")
        if user_text.lower() == "q":
            break

        label, confidence = predict_text(model, user_text)
        print("Prediction:", label)
        print("Confidence:", confidence)
