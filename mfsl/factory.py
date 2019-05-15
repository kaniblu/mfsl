import itertools
import collections

from . import parser
from .utils import to_string


class FeatureFactory(object):

    def __init__(self, feature_classes):
        self.parser = parser.Parser()
        self.feature_classes = feature_classes

        self.feature_dict = {f.name: f for f in self.feature_classes}
        self.feature_cache = dict()

    def _parse_from(self, data):
        assert "feat" in data and "kwargs" in data, \
            f"incorrect data format: {data}"

        name = data["feat"]
        args, kwargs = data.get("args", tuple()), data.get("kwargs", dict())

        assert name in self.feature_dict, \
            (f"unrecognized feature name: {name} (did you update "
             f"`create_factory` function in `features.py`?)")

        cls = self.feature_dict[name]
        for v in itertools.chain(kwargs.values(), args):
            vtype = v["type"]
            if vtype == "feature":
                v["value"] = self.parse_from(v["value"])

        args = [v["value"] for v in args]
        kwargs = {k: v["value"] for k, v in kwargs.items()}

        ret = cls(*args, **kwargs)
        ret.factory = self
        return ret

    def parse_from(self, string):
        if isinstance(string, collections.Mapping):
            string = to_string(string)

        p = self.parser.parse(string)
        if p is None:
            raise ValueError("not a valid feature serialization")
        feat = self._parse_from(p)

        if str(feat) not in self.feature_cache:
            self.feature_cache[str(feat)] = feat

        return self.feature_cache[str(feat)]


