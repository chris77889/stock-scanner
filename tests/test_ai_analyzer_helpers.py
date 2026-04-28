import pytest

from services.ai_analyzer import AIAnalyzer


@pytest.fixture
def analyzer():
    return AIAnalyzer(
        custom_api_url="https://example.com/v1/chat/completions",
        custom_api_key="test-key",
        custom_api_model="test-model",
        custom_api_timeout=5,
    )


def test_extract_recommendation_returns_buy(analyzer):
    text = "## 投资建议\n建议买入并逐步增持。\n"
    assert analyzer._extract_recommendation(text) == "买入"


def test_extract_recommendation_returns_sell(analyzer):
    text = "## 投资建议\n当前位置建议减持，避免进一步回撤。\n"
    assert analyzer._extract_recommendation(text) == "卖出"


def test_extract_recommendation_returns_hold(analyzer):
    text = "## 投资建议\n短期以持有为主，等待方向确认。\n"
    assert analyzer._extract_recommendation(text) == "持有"


def test_extract_recommendation_falls_back_to_watch(analyzer):
    text = "分析内容里没有结构化建议。"
    assert analyzer._extract_recommendation(text) == "观望"


def test_calculate_analysis_score_rewards_bullish_text(analyzer):
    technical_summary = {
        "trend": "upward",
        "volume_trend": "increasing",
        "rsi_level": 55,
    }

    score = analyzer._calculate_analysis_score("整体看涨，建议买入。", technical_summary)

    assert score == 75


def test_calculate_analysis_score_penalizes_bearish_text(analyzer):
    technical_summary = {
        "trend": "downward",
        "volume_trend": "decreasing",
        "rsi_level": 75,
    }

    score = analyzer._calculate_analysis_score("走势偏弱，看跌，建议卖出。", technical_summary)

    assert score == 10
