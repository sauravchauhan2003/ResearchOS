from pathlib import Path

from memory.memory_manager import MemoryManager
from memory.models import MEMORY_SCHEMA_VERSION
from memory.pickle_store import PickleStore


SIGNING_KEY = b"test-only-signing-key-for-memory-tests-0001"


def manager(tmp_path: Path, account: str = "account-a") -> MemoryManager:
    return MemoryManager(tmp_path / account, tmp_path / "signing.key", signing_key=SIGNING_KEY)


def test_memory_crud_search_and_restart_persistence(tmp_path: Path) -> None:
    first = manager(tmp_path)
    record = first.add(
        user_query="Compare battery recycling methods",
        findings="Hydrometallurgy can recover lithium and nickel.",
        claims=[{"text": "Recovery rates depend on feedstock", "sources": ["https://example.test/source"]}],
        conclusions="Compare process costs with environmental impact.",
        unresolved_questions=["What is the commercial recovery rate?"],
        user_preferences={"citation_style": "links"},
    )
    restarted = manager(tmp_path)
    assert restarted.get(record.memory_id) == record
    assert restarted.search("lithium recovery") == [record]
    changed = restarted.update(record.memory_id, conclusions="Evidence needs a regional cost comparison.")
    assert changed.conclusions.startswith("Evidence needs")
    assert restarted.delete(record.memory_id)
    assert restarted.get(record.memory_id) is None


def test_account_memories_are_isolated(tmp_path: Path) -> None:
    manager(tmp_path, "account-a").add(user_query="Private query about topic alpha")
    manager(tmp_path, "account-b").add(user_query="Private query about topic beta")
    assert len(manager(tmp_path, "account-a").search()) == 1
    assert len(manager(tmp_path, "account-b").search()) == 1
    assert "beta" not in manager(tmp_path, "account-a").search()[0].user_query


def test_corrupt_file_is_preserved_and_replaced_with_empty_schema(tmp_path: Path) -> None:
    current = manager(tmp_path)
    current.store.path.parent.mkdir(parents=True, exist_ok=True)
    current.store.path.write_bytes(b"not a trusted pickle")
    assert current.search() == []
    assert current.last_warning and "preserved" in current.last_warning
    preserved = list(current.store.path.parent.glob("*.corrupt-*"))
    assert len(preserved) == 1
    current.add(user_query="new safe entry")
    payload = PickleStore(current.store.path, tmp_path / "signing.key", SIGNING_KEY)._decode(current.store.path.read_bytes())
    assert payload["schema_version"] == MEMORY_SCHEMA_VERSION
    assert payload["records"][0]["user_query"] == "new safe entry"
