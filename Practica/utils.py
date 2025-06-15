import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

def clean_text(text):
    """Basic text cleaner: lowercase, remove punctuation and extra whitespace"""
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)  # remove punctuation
    text = re.sub(r"\s+", " ", text).strip()
    return text

def prepare_data(data, target, test_size=0.2, max_features=5000):
    """Cleans and vectorizes text data using TF-IDF"""
    df = pd.DataFrame({"text": data, "target": target})
    df["clean_text"] = df["text"].apply(clean_text)

    vectorizer = TfidfVectorizer(max_features=max_features)
    X = vectorizer.fit_transform(df["clean_text"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    return X_train, X_test, y_train, y_test, vectorizer
