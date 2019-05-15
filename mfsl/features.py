import logging
import inspect
import collections

import numpy as np
import pandas as pd
import stockstats

from .utils import to_string
from .utils import roll
from .common import parse_feature


class AbstractFeature(object):
    """
    Abstract class for feature specification
    All inherting feature classes must at least specify the name (e.g. 'ma')
    and the minimum set of keyword arguments (an ordered dictionary that maps
    argument names to default values)
    """
    name = None
    kwargs = collections.OrderedDict()

    def __init__(self, *args, **kwargs):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.factory = None
        self.cache = dict()
        filled = dict()

        for k, v in zip(self.kwargs, args):
            filled[k] = v

        for k, v in kwargs.items():
            if k not in kwargs:
                self.logger.warning(f"unrecognized argument: {k}")
                continue
            if k in filled:
                self.logger.warning(f"key already filled by "
                                    f"positional arguments: {k}")
            filled[k] = v

        for k, v in self.kwargs.items():
            if k in filled:
                continue
            self.logger.info(f"'{k}' has not been filled; setting default "
                             f"value '{v}'")
            filled[k] = v

        for k, v in filled.items():
            setattr(self, k, v)

    def to_data(self):
        return {
            "feat": self.name,
            "args": [],
            "kwargs": collections.OrderedDict([
                (k, {
                    "type": type(default),
                    "value": getattr(self, k, default)
                })
                for k, default in self.kwargs.items()
            ])
        }

    def __str__(self):
        return to_string(self.to_data())

    def extract(self, df: pd.DataFrame) -> pd.Series:
        """
        Extract the feature from the stock DataFrame.
        The stock DataFrame should at least have columns ('timestamp',
        'volume', ...), and the datetime column 'timestamp' must be the index.
        :param df:
        :return: pd.Series
        """
        raise NotImplementedError()

    def __call__(self, df: pd.DataFrame) -> pd.Series:
        """
        Warning!
        This method assumes that the dataframe is immutable: unique identifier
        (memory address) will be used as the key for caching.
        """
        if id(df) not in self.cache:
            feat = self.extract(df)
            dtype = feat.dtype

            if dtype.name.startswith("float"):
                if feat.isna().sum() > 0:
                    feat.fillna(method="backfill", inplace=True)
                feat = (feat + 1e-7).astype(dtype, copy=False)

            self.cache[id(df)] = feat
        return self.cache[id(df)]


class BollingerUpperFeature(AbstractFeature):
    """
    Bollinger Bands
    Bollinger Bands are in fact a triple of indicators that consists of moving
    averages and windowed standard deviations, calculated by the following
    formulas:
     - Middle: simple moving average (SMA) with window size N
     - Upper: SMA with window size N + K x N-period standard deviation
     - Lower: SMA with window size N - K x N-period standard deviation
    """

    name = "bollu"
    kwargs = collections.OrderedDict([
        ("feat", None),
        ("k", 2),
        ("win", 20),
    ])

    def extract(self, df: pd.DataFrame):
        if self.feat is None:
            self.feat = parse_feature("col(lhc)")

        sma = parse_feature(f"ma({self.feat}, {self.win})")
        std = parse_feature(f"std({self.feat}, {self.win})")
        return sma(df) + self.k * std(df)


class BollingerLowerFeature(AbstractFeature):
    """
    Bollinger Bands
    Bollinger Bands are in fact a triple of indicators that consists of moving
    averages and windowed standard deviations, calculated by the following
    formulas:
     - Middle: simple moving average (SMA) with window size N
     - Upper: SMA with window size N + K x N-period standard deviation
     - Lower: SMA with window size N - K x N-period standard deviation
    """

    name = "bollb"
    kwargs = collections.OrderedDict([
        ("feat", None),
        ("k", 2),
        ("win", 20),
    ])

    def extract(self, df: pd.DataFrame):
        if self.feat is None:
            self.feat = parse_feature("col(lhc)")

        sma = parse_feature(f"ma({self.feat}, {self.win})")
        std = parse_feature(f"std({self.feat}, {self.win})")
        return sma(df) - self.k * std(df)


class BollingerBandwithFeature(AbstractFeature):
    """
    Bollinger Bands Bandwidth
    """

    name = "bollw"
    kwargs = collections.OrderedDict([
        ("feat", AbstractFeature()),
        ("k", 2),
        ("win", 20),
    ])

    def extract(self, df: pd.DataFrame):
        bu = parse_feature(f"bollu({self.feat}, {self.k}, {self.win})")
        bl = parse_feature(f"bollb({self.feat}, {self.k}, {self.win})")
        bm = parse_feature(f"ma({self.feat}, {self.win})")
        return (bu(df) - bl(df)) / bm(df)


class BollingerPercentageFeature(AbstractFeature):
    """
    Bollinger Bands Indicators
    Bollinger Bands Indicators are derived from Bollinger Bands.
        (feat - bollb + eps) / (bollu - bollb + eps)
    """

    name = "bollp"
    kwargs = collections.OrderedDict([
        ("feat", AbstractFeature()),
        ("bfeat", None),
        ("k", 2),
        ("win", 20),
        ("eps", 0.1),
    ])

    def extract(self, df: pd.DataFrame):
        if self.bfeat is None:
            self.bfeat = parse_feature("col(lhc)")
        dtype = self.bfeat(df).dtype
        bu = parse_feature(f"bollu({self.bfeat}, {self.k}, {self.win})")
        bl = parse_feature(f"bollb({self.bfeat}, {self.k}, {self.win})")
        bp = (self.feat(df) - bl(df) + self.eps) / (bu(df) - bl(df) + self.eps)
        return bp.astype(dtype, copy=False)


class ColumnFeature(AbstractFeature):
    name = "col"
    kwargs = collections.OrderedDict([
        ("col", None)
    ])

    def extract(self, df: pd.DataFrame):
        assert self.col is not None, \
            "column name must be supplied"
        return df[self.col]


class DeltaFeature(AbstractFeature):
    name = "delta"
    kwargs = collections.OrderedDict([
        ("source", AbstractFeature()),
        ("target", AbstractFeature()),
        ("ratio", True)
    ])

    def extract(self, df: pd.DataFrame):
        delta = self.target(df) - self.source(df)
        if self.ratio:
            return (delta / self.source(df))
        else:
            return delta


class DiscretizedFeature(AbstractFeature):
    name = "bin"
    kwargs = collections.OrderedDict([
        ("feat", AbstractFeature()),
        ("bins", "0")
    ])

    def __init__(self, *args, **kwargs):
        super(DiscretizedFeature, self).__init__(*args, **kwargs)
        self.bins = str(self.bins)
        self.bins_list = list(map(float, self.bins.split(",")))

    def get_bounds(self, idx):
        """
        returns the boundary of given discretized index.
        :param idx (int): discretized index
        :return: a tuple of int or float. A None value indicates negative or
                 positive infinity.
        """
        if not (0 <= idx <= len(self.bins_list)):
            raise ValueError(f"index must be between 0 and "
                             f"{len(self.bins_list)}. current index: {idx}")
        bounds = [None] + list(self.bins_list) + [None]
        return (bounds[idx], bounds[idx + 1])

    def extract(self, df: pd.DataFrame):
        bins = [float("-inf")] + self.bins_list + [float("inf")]
        return pd.cut(self.feat(df), bins=bins)


class LogFeature(AbstractFeature):
    name = "log"
    kwargs = collections.OrderedDict([
        ("feat", AbstractFeature())
    ])

    def extract(self, df: pd.DataFrame):
        return np.log(self.feat(df))


class MarketIndexFeature(AbstractFeature):
    name = "index"
    kwargs = collections.OrderedDict([
        ("index", None)
    ])

    def extract(self, df: pd.DataFrame):
        assert self.index is not None, \
            "An index name must be provided"

        df = stockstats.StockDataFrame.retype(df)
        return df[self.index]


class MovingAverageFeature(AbstractFeature):
    name = "ma"
    kwargs = collections.OrderedDict([
        ("feat", AbstractFeature()),
        ("win", 15)
    ])

    def extract(self, df: pd.DataFrame):
        return roll(self.feat(df), self.win, lambda x: x.mean())


class OffsetFeature(AbstractFeature):
    name = "shift"
    kwargs = collections.OrderedDict([
        ("feat", AbstractFeature()),
        ("n", 1),
        # ("fill", "interpolate")
    ])

    def extract(self, df: pd.DataFrame):
        assert isinstance(self.n, (int, str)), \
            "offset size must be either an integer or a timedelta string"
        # utils.assert_oneof(self.fill, {"pad", "zero", "interpolate", })

        if isinstance(self.n, int):
            assert self.n >= 0, \
                "offset size must be larger than or equal to 0"

        feat = self.feat(df)
        if isinstance(self.n, int):
            feat = feat.shift(self.n)
            feat = feat.fillna(method="backfill")
        elif isinstance(self.n, str):
            # new_idx = feat.index - pd.to_timedelta(self.offset)
            # other = pd.DataFrame(index=new_idx)
            raise NotImplementedError()

        return feat


class RescaleFeature(AbstractFeature):
    name = "rescale"
    kwargs = collections.OrderedDict([
        ("feat", AbstractFeature()),
        ("amin", -1),
        ("amax", 1),
        ("pmin", None),
        ("pmax", None)
    ])

    def extract(self, df: pd.DataFrame):
        feat = self.feat(df)
        amin, amax, pmin, pmax = self.amin, self.amax, self.pmin, self.pmax
        if pmin is None:
            pmin = feat.min()
        if pmax is None:
            pmax = feat.max()
        norm = (feat - pmin) / (pmax - pmin)
        return norm * (amax - amin) + amin


class RSIFeature(AbstractFeature):
    name = "rsi"
    kwargs = collections.OrderedDict([
        ("win", 6)
    ])

    def extract(self, df: pd.DataFrame):
        assert isinstance(self.win, int) and self.win >= 0, \
            f"window size must be an integer and larger than 0: {self.win}"
        return parse_feature(f"index(rsi_{self.win})")(df)


class StandardDeviationFeature(AbstractFeature):
    name = "std"
    kwargs = collections.OrderedDict([
        ("feat", AbstractFeature()),
        ("win", 6)
    ])

    def extract(self, df: pd.DataFrame):
        assert isinstance(self.win, int) and self.win >= 0, \
            f"window size must be an integer and larger than 0: {self.win}"

        return roll(self.feat(df), self.win, lambda x: x.std())


class StochasticRSIFeature(AbstractFeature):
    """
    Stochastic RSI
    This is essentially window-normalized percentage of the RSI indicator.
    More info: https://www.tradingview.com/wiki/Stochastic_RSI_(STOCH_RSI)
    """

    name = "srsi"
    kwargs = collections.OrderedDict([
        ("win", 6)
    ])

    def extract(self, df: pd.DataFrame):
        assert isinstance(self.win, int) and self.win >= 0, \
            f"window size must be an integer and larger than 0: {self.win}"

        rsi = parse_feature(f"rsi({self.win})")(df)
        rsi_min = roll(rsi, self.win, lambda x: x.min())
        rsi_max = roll(rsi, self.win, lambda x: x.max())
        return (rsi - rsi_min) / (rsi_max - rsi_min)


class WeightedMovingAverageFeature(AbstractFeature):
    name = "wma"
    kwargs = collections.OrderedDict([
        ("feat", AbstractFeature()),
        ("win", 15),
        ("method", "exp")
    ])

    def __init__(self, *args, **kwargs):
        super(WeightedMovingAverageFeature, self).__init__(*args, **kwargs)
        self._expweight_cache = dict()

    @staticmethod
    def softmax(x):
        x -= x.min()
        exp = np.exp(x)
        return exp / exp.sum()

    def extract(self, df: pd.DataFrame):
        assert isinstance(self.win, int) and self.win > 0, \
            "window size must be an integer and larger than 0"

        feat = self.feat(df)

        if self.method == "exp":
            weights = self.softmax(np.arange(self.win))

            def weighted_mean(x):
                return (x * weights[:len(x)]).sum()

            return roll(feat, self.win, lambda x: x.apply(weighted_mean))
        else:
            raise ValueError(f"unrecognized method: {self.method}")


def feature_classes():
    return [cls for _, cls in globals().items()
            if inspect.isclass(cls) and issubclass(cls, AbstractFeature)]
