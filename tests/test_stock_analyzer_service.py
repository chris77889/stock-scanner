import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pandas as pd
import pytest

from services.stock_analyzer_service import StockAnalyzerService



def build_analyzer_dataframe():
    return pd.DataFrame(
        [
            {
                "Close": 10.0,
                "Change": 0.1,
                "Change_pct": 1.0,
                "MA5": 9.0,
                "MA20": 8.0,
                "MA60": 7.0,
                "MACD": 1.0,
                "Signal": 0.5,
                "Volume": 1000,
                "Volume_MA": 800,
                "RSI": 55.0,
                "MACD_Signal": 0.4,
                "Volume_Ratio": 1.2,
                "Volatility": 2.5,
            },
            {
                "Close": 11.0,
                "Change": 0.2,
                "Change_pct": 2.0,
                "MA5": 10.0,
                "MA20": 9.0,
                "MA60": 8.0,
                "MACD": 1.2,
                "Signal": 0.6,
                "Volume": 2000,
                "Volume_MA": 1000,
                "RSI": 60.0,
                "MACD_Signal": 0.5,
                "Volume_Ratio": 2.0,
                "Volatility": 3.0,
            },
        ]
    )


async def collect_async_chunks(async_iterable):
    items = []
    async for item in async_iterable:
        items.append(json.loads(item))
    return items


async def fake_ai_chunks():
    yield json.dumps({"stock_code": "000001", "status": "analyzing", "ai_analysis_chunk": "AI片段"})
    yield json.dumps({"stock_code": "000001", "status": "completed", "score": 88, "recommendation": "买入"})


@pytest.fixture
def service(monkeypatch):
    monkeypatch.setattr("services.stock_analyzer_service.StockDataProvider", lambda: SimpleNamespace())
    monkeypatch.setattr("services.stock_analyzer_service.TechnicalIndicator", lambda: SimpleNamespace())
    monkeypatch.setattr("services.stock_analyzer_service.StockScorer", lambda: SimpleNamespace())
    monkeypatch.setattr("services.stock_analyzer_service.AIAnalyzer", lambda **kwargs: SimpleNamespace())
    return StockAnalyzerService()


@pytest.mark.asyncio
async def test_analyze_stock_returns_error_for_dataframe_with_error(service):
    error_df = pd.DataFrame()
    error_df.error = "mocked upstream error"

    service.data_provider.get_stock_data = AsyncMock(return_value=error_df)

    results = await collect_async_chunks(service.analyze_stock("000001"))

    assert results == [
        {
            "stock_code": "000001",
            "market_type": "A",
            "error": "mocked upstream error",
            "status": "error",
        }
    ]


@pytest.mark.asyncio
async def test_analyze_stock_returns_error_for_empty_dataframe(service):
    service.data_provider.get_stock_data = AsyncMock(return_value=pd.DataFrame())

    results = await collect_async_chunks(service.analyze_stock("000001"))

    assert results == [
        {
            "stock_code": "000001",
            "market_type": "A",
            "error": "获取到的股票 000001 数据为空",
            "status": "error",
        }
    ]


@pytest.mark.asyncio
async def test_analyze_stock_yields_basic_result_before_ai_chunks(service):
    df = build_analyzer_dataframe()
    service.data_provider.get_stock_data = AsyncMock(return_value=df)
    service.indicator.calculate_indicators = lambda value: value
    service.scorer.calculate_score = lambda value: 72
    service.scorer.get_recommendation = lambda score: "推荐"
    service.ai_analyzer.get_ai_analysis = lambda *args, **kwargs: fake_ai_chunks()

    results = await collect_async_chunks(service.analyze_stock("000001", stream=True))

    assert results[0]["stock_code"] == "000001"
    assert results[0]["score"] == 72
    assert results[0]["recommendation"] == "推荐"
    assert results[0]["ma_trend"] == "UP"
    assert results[0]["macd_signal"] == "BUY"
    assert results[0]["volume_status"] == "HIGH"
    assert results[1] == {"stock_code": "000001", "status": "analyzing", "ai_analysis_chunk": "AI片段"}
    assert results[2] == {"stock_code": "000001", "status": "completed", "score": 88, "recommendation": "买入"}


@pytest.mark.asyncio
async def test_scan_stocks_emits_ranked_results_and_completion(service):
    df = build_analyzer_dataframe()
    service.data_provider.get_multiple_stocks_data = AsyncMock(return_value={"000001": df, "000002": df})
    service.indicator.calculate_indicators = lambda value: value
    service.scorer.batch_score_stocks = lambda stock_map: [
        ("000001", 80, "强烈推荐"),
        ("000002", 50, "观望"),
    ]
    service.ai_analyzer.get_ai_analysis = lambda *args, **kwargs: fake_ai_chunks()

    results = await collect_async_chunks(service.scan_stocks(["000001", "000002"], min_score=60, stream=False))

    assert results[0] == {
        "stream_type": "batch",
        "stock_codes": ["000001", "000002"],
        "market_type": "A",
        "min_score": 60,
    }
    assert results[1]["stock_code"] == "000001"
    assert results[1]["status"] == "waiting"
    assert results[2]["stock_code"] == "000002"
    assert results[2]["status"] == "completed"
    assert results[-1] == {
        "scan_completed": True,
        "total_scanned": 2,
        "total_matched": 1,
    }
