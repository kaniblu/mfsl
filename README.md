# MFSL (Market Feature Specification Language) #

This miniature library implements a basic toolkit for specifying and extracting
market features from a simple [olhcv](https://en.wikipedia.org/wiki/Open-high-low-close_chart) market data.

This package can be installed using `setuptools`.

    python setup.py install


## Quickstart ##

All market indicators are derived from the select core market features, namely
starting price (`open`), closing price (`close`), maximum price (`high`), 
minimum price (`low`) and volume within a fixed interval. 

For example, [MACD](https://www.investopedia.com/terms/m/macd.asp) is a 
frequently used market indicator that is derived from the difference ratio between
one moving average to another.

This library toolkit is meant to make creating customized market indicators
easier by providing common market time-series operators through the specification language.

For example, the most common form of MACD can be expressed in terms of MFSL as 

    delta(source=wma(col(close), 26), 
          target=wma(col(close), 12), ratio=true)

The formula above specifies the difference ratio (`delta`) of
the exponential moving average of `close` of window size 12 from the exponential
moving average of `close` of window size 26.

There could be many variants of MACD where the moving average window sizes 
could be different, or the source EMA could be lagged behind by 10 time-steps, 
such as the following example.

    delta(source=shift(wma(col(close), 34), n=10), 
          target=wma(col(close), 12), ratio=true)

Many more market indicators can be created by combining pre-defined indicators,
such as [RSI](https://www.investopedia.com/terms/r/rsi.asp) (`rsi`), 
[SRSI](https://www.investopedia.com/terms/s/stochrsi.asp) (`srsi`), 
[Bollinger Bandwidth](https://www.investopedia.com/terms/b/bollingerbands.asp) (`bollw`), etc.

To apply feature specifications on a specific market data, first, create a
feature extractor:

    >>> import mfsl
    >>> feat = mfsl.parse_feature("delta(source=wma(col(close), 26),            "
                                  "      target=wma(col(close), 12), ratio=true)")

MFSL assumes that the market data is stored in `pandas.DataFrame` with the same
format as the following example data:

    >>> data
                               volume      open      high       low     close
    timestamp
    2018-11-08 09:52:00  1.200376e+00  0.000524  0.000524  0.000524  0.000524
    2018-11-08 09:54:00  1.814574e+00  0.000524  0.000524  0.000524  0.000524
    2018-11-08 09:56:00  1.110756e+00  0.000524  0.000524  0.000524  0.000524
    2018-11-08 10:01:00  6.229694e+01  0.000524  0.000525  0.000524  0.000525
    2018-11-08 10:05:00  1.267060e+01  0.000521  0.000521  0.000521  0.000521
    2018-11-08 10:08:00  5.060300e+01  0.000522  0.000522  0.000522  0.000522
    2018-11-08 10:12:00  7.213635e+00  0.000525  0.000525  0.000525  0.000525
    2018-11-08 10:13:00  4.636602e+02  0.000525  0.000525  0.000525  0.000525
    2018-11-08 10:19:00  1.635157e+02  0.000525  0.000525  0.000525  0.000525
    2018-11-08 10:20:00  5.611069e+02  0.000527  0.000527  0.000527  0.000527
    2018-11-08 10:30:00  2.610119e+00  0.000525  0.000525  0.000525  0.000525
    2018-11-08 10:31:00  3.446510e+00  0.000527  0.000527  0.000527  0.000527
    ...
    
    [2650 rows x 5 columns]

Then simply apply the feature extractor on the `DataFrame`, which returns a
`pandas.Series` object as the result:

    >>> feat(data)
    timestamp
    2018-11-08 09:52:00    5.535647e-02
    2018-11-08 09:54:00    2.058278e-01
    2018-11-08 09:56:00    6.148511e-01
    2018-11-08 10:01:00    1.727454e+00
    2018-11-08 10:05:00    4.733711e+00
    2018-11-08 10:08:00    1.291270e+01
    2018-11-08 10:12:00    3.527695e+01
    2018-11-08 10:13:00    9.606509e+01
    2018-11-08 10:19:00    2.612733e+02
    2018-11-08 10:20:00    7.115023e+02
    2018-11-08 10:30:00    1.929917e+03
    2018-11-08 10:31:00    5.240993e+03
    2018-11-08 10:38:00    5.179350e+03
    2018-11-08 10:43:00    5.069178e+03
    2018-11-08 10:44:00    4.807782e+03
    ...
    
    Name: close, Length: 2650, dtype: float64
