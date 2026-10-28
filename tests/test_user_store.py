from pathlib import Path

import pytest

from auth.user_store import UserStore
from config.settings import AppSettings


def test_register_authenticate_and_persisted_preferences(tmp_path: Path) -> None:
    db_path = tmp_path / "accounts.sqlite3"
    store = UserStore(db_path)
    user = store.register("saurav", "long-and-unique-password")

    assert store.authenticate("SAURAV", "long-and-unique-password") == user
    assert store.authenticate("saurav", "wrong-password") is None
    store.save_settings(user.id, {"provider": "Groq", "model": "a-live-model-id"})
    assert UserStore(db_path).get_settings(user.id) == {"provider": "Groq", "model": "a-live-model-id"}


def test_registration_rejects_weak_duplicate_and_invalid_usernames(tmp_path: Path) -> None:
    store = UserStore(tmp_path / "accounts.sqlite3")
    store.register("researcher", "correct-horse-battery")
    with pytest.raises(ValueError, match="already registered"):
        store.register("RESEARCHER", "another-long-password")
    with pytest.raises(ValueError, match="12 characters"):
        store.register("otheruser", "short")
    with pytest.raises(ValueError, match="Username"):
        store.register("not a username", "another-long-password")


def test_browser_login_sessions_are_hashed_expiring_and_revocable(tmp_path: Path) -> None:
    store = UserStore(tmp_path / "accounts.sqlite3")
    user = store.register("researcher", "correct-horse-battery")

    token = store.create_login_session(user.id)
    assert store.authenticate_login_session(token) == user
    with store._connect() as connection:
        saved_hash = connection.execute("SELECT token_hash FROM login_sessions").fetchone()[0]
    assert saved_hash != token

    store.revoke_login_session(token)
    assert store.authenticate_login_session(token) is None


def test_user_data_directories_are_separated_and_ids_are_validated(tmp_path: Path) -> None:
    settings = AppSettings(tmp_path, None, None)
    first = settings.user_data_dir("9206d759-6d68-4d95-b5e8-60c8ed266b5b")
    second = settings.user_data_dir("b9ae6e72-8bce-4bb3-a59d-6eb7fd5c1a35")
    assert first != second
    with pytest.raises(ValueError):
        settings.user_data_dir("../../other-account")
