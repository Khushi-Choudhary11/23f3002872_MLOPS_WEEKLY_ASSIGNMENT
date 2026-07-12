import pandas as pd

DATA_PATH = "data/raw/iris.csv"


def test_dataset_exists():
    df = pd.read_csv(DATA_PATH)
    assert len(df) > 0


def test_no_missing_values():
    df = pd.read_csv(DATA_PATH)
    assert df.isnull().sum().sum() == 0


def test_columns():
    df = pd.read_csv(DATA_PATH)

    expected_columns = [
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width",
        "species",
    ]

    assert list(df.columns) == expected_columns


def test_numeric_features():
    df = pd.read_csv(DATA_PATH)

    numeric_columns = [
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width",
    ]

    for col in numeric_columns:
        assert pd.api.types.is_numeric_dtype(df[col])


def test_value_ranges():
    df = pd.read_csv(DATA_PATH)

    assert df["sepal_length"].between(4, 8).all()
    assert df["sepal_width"].between(2, 5).all()
    assert df["petal_length"].between(1, 7).all()
    assert df["petal_width"].between(0, 3).all()
