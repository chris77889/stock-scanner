from utils.api_utils import APIUtils


def test_format_api_url_with_trailing_slash():
    assert APIUtils.format_api_url("https://example.com/") == "https://example.com/chat/completions"


def test_format_api_url_with_hash_suffix():
    assert APIUtils.format_api_url("https://example.com/custom#") == "https://example.com/custom"


def test_format_api_url_with_plain_host():
    assert APIUtils.format_api_url("https://example.com") == "https://example.com/v1/chat/completions"


def test_format_api_url_with_empty_value():
    assert APIUtils.format_api_url("") == ""
