import joblib
import pandas as pd

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score

MODEL_PATH = "models/model.joblib"
DATA_PATH = "data/raw/iris.csv"


def test_model_exists():
    model = joblib.load(MODEL_PATH)
    assert model is not None


def test_model_accuracy():
    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=["species"])
    y = df["species"]

    model = joblib.load(MODEL_PATH)

    predictions = model.predict(X)

    accuracy = accuracy_score(y, predictions)

    assert accuracy >= 0.90


def test_model_precision():
    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=["species"])
    y = df["species"]

    model = joblib.load(MODEL_PATH)

    predictions = model.predict(X)

    precision = precision_score(
        y,
        predictions,
        average="weighted"
    )

    assert precision >= 0.90
