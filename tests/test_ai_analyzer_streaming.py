import json

import pandas as pd
import pytest

from services.ai_analyzer import AIAnalyzer


class MockStreamResponse:
    def __init__(self, chunks, status_code=200):
        self._chunks = chunks
        self.status_code = status_code

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def aiter_text(self):
        for chunk in self._chunks:
            yield chunk

    async def aread(self):
        return b'{"error": {"message": "mocked error"}}'


class MockAsyncClient:
    def __init__(self, *args, **kwargs):
        self.stream_response = kwargs.pop("stream_response")
        self.post_response = kwargs.pop("post_response", None)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def stream(self, method, url, json=None, headers=None):
        return self.stream_response

    async def post(self, url, json=None, headers=None):
        return self.post_response


class MockPostResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def json(self):
        return self._payload


def build_dataframe():
    rows = []
    for idx in range(14):
        rows.append(
            {
                "RSI": 55 + idx * 0.1,
                "Close": 100 + idx,
                "Change": 1.2,
                "MA5": 105 + idx,
                "MA20": 100 + idx,
                "MACD": 1.5,
                "MACD_Signal": 1.0,
                "Volume_Ratio": 1.1,
                "Volatility": 2.5,
            }
        )
    return pd.DataFrame(rows)


async def collect_stream_results(chunks, monkeypatch):
    stream_response = MockStreamResponse(chunks)

    def mock_async_client(*args, **kwargs):
        return MockAsyncClient(*args, stream_response=stream_response, **kwargs)

    monkeypatch.setattr("services.ai_analyzer.httpx.AsyncClient", mock_async_client)

    analyzer = AIAnalyzer(
        custom_api_url="https://example.com/v1/chat/completions",
        custom_api_key="test-key",
        custom_api_model="test-model",
        custom_api_timeout=5,
    )

    results = []
    async for item in analyzer.get_ai_analysis(build_dataframe(), "000001", stream=True):
        results.append(json.loads(item))
    return results


async def collect_non_stream_results(payload, monkeypatch, status_code=200):
    post_response = MockPostResponse(payload, status_code=status_code)

    def mock_async_client(*args, **kwargs):
        return MockAsyncClient(*args, stream_response=None, post_response=post_response, **kwargs)

    monkeypatch.setattr("services.ai_analyzer.httpx.AsyncClient", mock_async_client)

    analyzer = AIAnalyzer(
        custom_api_url="https://example.com/v1/chat/completions",
        custom_api_key="test-key",
        custom_api_model="test-model",
        custom_api_timeout=5,
    )

    results = []
    async for item in analyzer.get_ai_analysis(build_dataframe(), "000001", stream=False):
        results.append(json.loads(item))
    return results


@pytest.mark.asyncio
async def test_streaming_handles_split_json_and_empty_choices(monkeypatch):
    chunks = [
        'data: {"choices":[{"delta":{"content":"Hel',
        'lo"}}]}\n',
        'data: {"choices":[]}\n',
        'data: {"choices":[{"delta":{"content":" World"}}]}\n',
        'data: [DONE]\n',
    ]

    results = await collect_stream_results(chunks, monkeypatch)

    assert [item["status"] for item in results] == ["analyzing", "analyzing", "analyzing", "analyzing", "completed"]
    assert [item.get("ai_analysis_chunk") for item in results if "ai_analysis_chunk" in item] == [
        "Hello",
        " World",
        "\n",
    ]
    assert results[-1]["status"] == "completed"


@pytest.mark.asyncio
async def test_streaming_flushes_trailing_buffer_without_newline(monkeypatch):
    chunks = [
        'data: {"choices":[{"delta":{"content":"Tail"}}]}',
    ]

    results = await collect_stream_results(chunks, monkeypatch)

    assert any(item.get("ai_analysis_chunk") == "Tail" for item in results)
    assert results[-1]["status"] == "completed"


@pytest.mark.asyncio
async def test_streaming_returns_error_chunk_when_top_level_error_present(monkeypatch):
    chunks = [
        'data: {"error":{"message":"upstream failed"}}\n',
    ]

    results = await collect_stream_results(chunks, monkeypatch)

    assert results[1] == {
        "stock_code": "000001",
        "error": "流式响应错误: {'message': 'upstream failed'}",
        "status": "error",
    }


@pytest.mark.asyncio
async def test_streaming_ignores_finish_reason_stop_without_crashing(monkeypatch):
    chunks = [
        'data: {"choices":[{"delta":{"content":"Part1"},"finish_reason":null}]}\n',
        'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n',
        'data: [DONE]\n',
    ]

    results = await collect_stream_results(chunks, monkeypatch)

    assert [item.get("ai_analysis_chunk") for item in results if "ai_analysis_chunk" in item] == [
        "Part1",
        "\n",
    ]
    assert results[-1]["status"] == "completed"


@pytest.mark.asyncio
async def test_non_streaming_handles_empty_choices_without_index_error(monkeypatch):
    results = await collect_non_stream_results({"choices": []}, monkeypatch)

    assert len(results) == 2
    assert results[0]["status"] == "analyzing"
    assert results[1]["status"] == "completed"
    assert results[1]["analysis"] == ""
