import pandas as pd

from services.stock_scorer import StockScorer



def make_df(**overrides):
    row = {
        "Close": 120,
        "MA5": 125,
        "MA20": 118,
        "MA60": 110,
        "RSI": 60,
        "MACD": 1.5,
        "Signal": 1.0,
        "Volume_Ratio": 1.8,
    }
    row.update(overrides)
    return pd.DataFrame([row])



def test_calculate_score_for_bullish_setup():
    scorer = StockScorer()

    score = scorer.calculate_score(make_df())

    assert score == 100



def test_calculate_score_for_weak_setup():
    scorer = StockScorer()

    score = scorer.calculate_score(
        make_df(
            Close=95,
            MA5=90,
            MA20=100,
            MA60=110,
            RSI=25,
            MACD=0.5,
            Signal=1.0,
            Volume_Ratio=0.8,
        )
    )

    assert score == 15



def test_get_recommendation_thresholds():
    scorer = StockScorer()

    assert scorer.get_recommendation(80) == "强烈推荐"
    assert scorer.get_recommendation(70) == "推荐"
    assert scorer.get_recommendation(60) == "谨慎推荐"
    assert scorer.get_recommendation(40) == "观望"
    assert scorer.get_recommendation(20) == "不推荐"
    assert scorer.get_recommendation(19) == "强烈不推荐"



def test_batch_score_stocks_sorts_descending_and_skips_invalid_frames():
    scorer = StockScorer()
    valid_high = make_df()
    valid_low = make_df(Close=95, MA5=90, MA20=100, MA60=110, RSI=25, MACD=0.5, Signal=1.0, Volume_Ratio=0.8)
    invalid = pd.DataFrame([{"Close": 100}])

    results = scorer.batch_score_stocks({
        "LOW": valid_low,
        "BROKEN": invalid,
        "HIGH": valid_high,
    })

    assert results == [
        ("HIGH", 100, "强烈推荐"),
        ("LOW", 15, "强烈不推荐"),
    ]
