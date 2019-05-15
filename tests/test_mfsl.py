import io
import gzip
import yaml

import mfsl
import pandas as pd


def load_data():
    with gzip.GzipFile("tests/stock.csv.gz", "r") as f:
        return pd.read_csv(
            io.TextIOWrapper(f),
            parse_dates=["timestamp"],
            index_col="timestamp"
        )


def load_features():
    with gzip.GzipFile("tests/features.yml.gz", "r") as f:
        return yaml.safe_load(io.TextIOWrapper(f))


def test_mfsl():
    data = load_data()
    features = load_features()
    features = {k: mfsl.parse_feature(f) for k, f in features.items()}
    for k, f in features.items():
        assert not f(data).isna().any()
