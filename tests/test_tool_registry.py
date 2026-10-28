import pytest

from tools.tool_registry import ToolRegistry


def test_registry_shares_tools_and_rejects_duplicate_names() -> None:
    search = object()
    registry = ToolRegistry(web_search=search)
    assert registry.get("WEB_SEARCH") is search
    assert registry.names() == ("web_search",)
    with pytest.raises(ValueError, match="already registered"):
        registry.register("web_search", object())
