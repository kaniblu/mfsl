import string
import itertools


def to_string(data):
    assert "feat" in data and "kwargs" in data, \
        f"incorrect data format: {data}"

    def escape(x: str):
        if x[0] not in string.ascii_letters:
            return f"'{x}'"
        return x

    def castr(x):
        if isinstance(x, bool):
            return "true" if x else "false"
        elif isinstance(x, str):
            return escape(x)
        else:
            return str(x)

    name = data["feat"]
    args, kwargs = data.get("args", tuple()), data.get("kwargs", dict())

    for v in itertools.chain(kwargs.values(), args):
        vtype = v["type"]
        if vtype == "feature":
            v["value"] = to_string(v["value"])

    args_str = ", ".join(castr(v["value"]) for v in args)
    kwargs_str = ", ".join(map("=".join, ((k, castr(v["value"]))
                                          for k, v in kwargs.items())))
    return f"{name}({', '.join(filter(None, (args_str, kwargs_str)))})"


def roll(s, window, func):
    dtype = s.dtype
    return func(s.rolling(window, min_periods=1)).astype(dtype)
