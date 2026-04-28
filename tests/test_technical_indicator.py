import pandas as pd

from services.technical_indicator import TechnicalIndicator



def build_price_dataframe(rows=80):
    data = []
    for idx in range(rows):
        close = 100 + idx
        data.append(
            {
                "Open": close - 1,
                "High": close + 2,
                "Low": close - 2,
                "Close": close,
                "Volume": 1000 + idx * 10,
            }
        )
    return pd.DataFrame(data)



def test_calculate_indicators_adds_expected_columns():
    indicator = TechnicalIndicator()
    df = build_price_dataframe()

    result = indicator.calculate_indicators(df)

    expected_columns = {
        "MA5",
        "MA20",
        "MA60",
        "RSI",
        "MACD",
        "Signal",
        "Histogram",
        "BB_Middle",
        "BB_Upper",
        "BB_Lower",
        "Volume_MA",
        "Volume_Ratio",
        "ATR",
        "Volatility",
    }

    assert expected_columns.issubset(result.columns)
    assert len(result) == len(df)
    assert result["MACD"].notna().any()
    assert result["Volume_Ratio"].notna().any()
