from agents.context import compact_blocks, compact_text


def test_compact_text_normalizes_whitespace_and_obeys_character_limit() -> None:
    compacted = compact_text("alpha\n\n beta  " + "long evidence " * 20, 80)
    assert len(compacted) <= 80
    assert "\n" not in compacted
    assert compacted.endswith("…[compacted]")


def test_compact_blocks_respects_shared_budget_and_keeps_each_block_prefix() -> None:
    blocks = [f"source-{index} " + "evidence " * 100 for index in range(5)]
    compacted = compact_blocks(blocks, max_chars=400, per_block_chars=200)
    assert len(compacted) <= 400
    for index in range(5):
        assert f"source-{index}" in compacted
