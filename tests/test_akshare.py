import pytest


pytestmark = pytest.mark.integration


def test_akshare_us_daily_fetches_data():
    ak = pytest.importorskip("akshare")
    df = ak.stock_us_daily(symbol="AAPL", adjust="qfq")

    assert df is not None
    assert not df.empty
