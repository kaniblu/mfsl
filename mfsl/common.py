import importlib

from .factory import FeatureFactory


def create_factory():
    features = importlib.import_module("..features", __name__)
    return FeatureFactory(features.feature_classes())


def parse_feature(s: str):
    global factory
    if "factory" not in globals():
        factory = create_factory()
    return factory.parse_from(s)
