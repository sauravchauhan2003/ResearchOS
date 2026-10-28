from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerifyMismatchError, VerificationError
from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from filelock import FileLock


USERNAME_RE = re.compile(r"^[a-zA-Z0-9_.-]{3,32}$")
API_KEY_PROVIDERS = {"Groq", "Gemini"}


@dataclass(frozen=True, slots=True)
class User:
    id: str
    username: str
    created_at: str


class UserStore:
    """SQLite account store; all future user data is keyed by this UUID."""

    def __init__(self, path: Path, *, password_hasher: PasswordHasher | None = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._api_key_encryption_key_path = self.path.with_name(".api-key-encryption-key")
        self._hasher = password_hasher or PasswordHasher()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 10000")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute(
                """CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY,
                    username TEXT NOT NULL UNIQUE COLLATE NOCASE,
                    password_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )"""
            )
            connection.execute(
                """CREATE TABLE IF NOT EXISTS user_settings (
                    user_id TEXT PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
                    settings_json TEXT NOT NULL DEFAULT '{}'
                )"""
            )
            connection.execute(
                """CREATE TABLE IF NOT EXISTS user_api_keys (
                    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    provider TEXT NOT NULL CHECK(provider IN ('Groq', 'Gemini')),
                    encrypted_key BLOB NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY(user_id, provider)
                )"""
            )
            connection.execute(
                """CREATE TABLE IF NOT EXISTS login_sessions (
                    token_hash TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    created_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL
                )"""
            )

    def create_login_session(self, user_id: str, *, remember: bool = True) -> str:
        """Create a revocable browser session; only its token hash is stored."""
        token = secrets.token_urlsafe(32)
        token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
        now = datetime.now(timezone.utc)
        expires_at = now + (timedelta(days=30) if remember else timedelta(hours=12))
        with self._connect() as connection:
            connection.execute("DELETE FROM login_sessions WHERE expires_at <= ?", (now.isoformat(),))
            exists = connection.execute("SELECT 1 FROM users WHERE id = ?", (user_id,)).fetchone()
            if exists is None:
                raise ValueError("Account not found.")
            connection.execute(
                "INSERT INTO login_sessions(token_hash, user_id, created_at, expires_at) VALUES (?, ?, ?, ?)",
                (token_hash, user_id, now.isoformat(), expires_at.isoformat()),
            )
        return token

    def authenticate_login_session(self, token: str) -> User | None:
        if not isinstance(token, str) or not 32 <= len(token) <= 256:
            return None
        token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as connection:
            row = connection.execute(
                "SELECT users.* FROM login_sessions JOIN users ON users.id = login_sessions.user_id "
                "WHERE login_sessions.token_hash = ? AND login_sessions.expires_at > ?",
                (token_hash, now),
            ).fetchone()
        return self._to_user(row) if row is not None else None

    def revoke_login_session(self, token: str | None) -> None:
        if not isinstance(token, str) or not 32 <= len(token) <= 256:
            return
        token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
        with self._connect() as connection:
            connection.execute("DELETE FROM login_sessions WHERE token_hash = ?", (token_hash,))

    def _api_key_encryption_key(self) -> bytes:
        """Load or create the server-local key used to encrypt account API keys."""
        path = self._api_key_encryption_key_path
        path.parent.mkdir(parents=True, exist_ok=True)
        with FileLock(str(path) + ".lock", timeout=15):
            try:
                key = path.read_bytes()
            except FileNotFoundError:
                key = secrets.token_bytes(32)
                try:
                    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
                except FileExistsError:
                    key = path.read_bytes()
                else:
                    with os.fdopen(descriptor, "wb") as target:
                        target.write(key)
                        target.flush()
                        os.fsync(target.fileno())
            if len(key) != 32:
                raise ValueError("The API-key encryption key is invalid; restore it or re-save account keys.")
            return key

    @staticmethod
    def _validate_api_key_provider(provider: str) -> str:
        if provider not in API_KEY_PROVIDERS:
            raise ValueError("Unsupported API-key provider.")
        return provider

    @staticmethod
    def _api_key_associated_data(user_id: str, provider: str) -> bytes:
        return f"ResearchOS API key:{user_id}:{provider}".encode("utf-8")

    def save_api_key(self, user_id: str, provider: str, api_key: str) -> None:
        provider = self._validate_api_key_provider(provider)
        value = api_key.strip()
        if not value:
            raise ValueError("Enter an API key before saving.")
        nonce = secrets.token_bytes(12)
        ciphertext = nonce + AESGCM(self._api_key_encryption_key()).encrypt(
            nonce, value.encode("utf-8"), self._api_key_associated_data(user_id, provider)
        )
        with self._connect() as connection:
            exists = connection.execute("SELECT 1 FROM users WHERE id = ?", (user_id,)).fetchone()
            if exists is None:
                raise ValueError("Account not found.")
            connection.execute(
                "INSERT INTO user_api_keys(user_id, provider, encrypted_key, updated_at) VALUES (?, ?, ?, ?) "
                "ON CONFLICT(user_id, provider) DO UPDATE SET "
                "encrypted_key = excluded.encrypted_key, updated_at = excluded.updated_at",
                (user_id, provider, ciphertext, datetime.now(timezone.utc).isoformat()),
            )

    def get_api_key(self, user_id: str, provider: str) -> str | None:
        provider = self._validate_api_key_provider(provider)
        with self._connect() as connection:
            row = connection.execute(
                "SELECT encrypted_key FROM user_api_keys WHERE user_id = ? AND provider = ?",
                (user_id, provider),
            ).fetchone()
        if row is None:
            return None
        encrypted = bytes(row["encrypted_key"])
        try:
            return AESGCM(self._api_key_encryption_key()).decrypt(
                encrypted[:12], encrypted[12:], self._api_key_associated_data(user_id, provider)
            ).decode("utf-8")
        except (InvalidTag, UnicodeDecodeError, ValueError) as exc:
            raise ValueError("This saved API key could not be decrypted. Remove it and save it again.") from exc

    def has_api_key(self, user_id: str, provider: str) -> bool:
        provider = self._validate_api_key_provider(provider)
        with self._connect() as connection:
            row = connection.execute(
                "SELECT 1 FROM user_api_keys WHERE user_id = ? AND provider = ?",
                (user_id, provider),
            ).fetchone()
        return row is not None

    def delete_api_key(self, user_id: str, provider: str) -> None:
        provider = self._validate_api_key_provider(provider)
        with self._connect() as connection:
            connection.execute(
                "DELETE FROM user_api_keys WHERE user_id = ? AND provider = ?",
                (user_id, provider),
            )

    @staticmethod
    def _to_user(row: sqlite3.Row) -> User:
        return User(row["id"], row["username"], row["created_at"])

    def register(self, username: str, password: str) -> User:
        normalized = username.strip()
        if not USERNAME_RE.fullmatch(normalized):
            raise ValueError("Username must be 3–32 letters, numbers, dots, underscores, or hyphens.")
        if len(password) < 12:
            raise ValueError("Password must contain at least 12 characters.")
        if len(password) > 1024:
            raise ValueError("Password is too long.")
        user = User(str(uuid.uuid4()), normalized, datetime.now(timezone.utc).isoformat())
        password_hash = self._hasher.hash(password)
        try:
            with self._connect() as connection:
                connection.execute(
                    "INSERT INTO users(id, username, password_hash, created_at) VALUES (?, ?, ?, ?)",
                    (user.id, user.username, password_hash, user.created_at),
                )
                connection.execute("INSERT INTO user_settings(user_id) VALUES (?)", (user.id,))
        except sqlite3.IntegrityError as exc:
            raise ValueError("That username is already registered.") from exc
        return user

    def authenticate(self, username: str, password: str) -> User | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM users WHERE username = ? COLLATE NOCASE", (username.strip(),)
            ).fetchone()
        if row is None:
            return None
        try:
            if not self._hasher.verify(row["password_hash"], password):
                return None
        except (VerifyMismatchError, VerificationError, InvalidHashError):
            return None
        if self._hasher.check_needs_rehash(row["password_hash"]):
            replacement = self._hasher.hash(password)
            with self._connect() as connection:
                connection.execute("UPDATE users SET password_hash = ? WHERE id = ?", (replacement, row["id"]))
        return self._to_user(row)

    def save_settings(self, user_id: str, settings: dict[str, object]) -> None:
        allowed = {"provider", "model", "key_modes"}
        if set(settings) - allowed:
            raise ValueError("Unsupported account setting.")
        key_modes = settings.get("key_modes")
        if key_modes is not None and (
            not isinstance(key_modes, dict)
            or any(provider not in API_KEY_PROVIDERS or mode not in {"shared", "personal"}
                   for provider, mode in key_modes.items())
        ):
            raise ValueError("Invalid provider key preference.")
        with self._connect() as connection:
            row = connection.execute(
                "SELECT settings_json FROM user_settings WHERE user_id = ?", (user_id,)
            ).fetchone()
            exists = connection.execute("SELECT 1 FROM users WHERE id = ?", (user_id,)).fetchone()
            if exists is None:
                raise ValueError("Account not found.")
            current = {}
            if row is not None:
                try:
                    decoded = json.loads(row["settings_json"])
                    current = decoded if isinstance(decoded, dict) else {}
                except (TypeError, json.JSONDecodeError):
                    current = {}
            previous_provider = current.get("provider")
            current.update(settings)
            if "provider" in settings and "model" not in settings and previous_provider != settings["provider"]:
                current.pop("model", None)
            connection.execute(
                "INSERT INTO user_settings(user_id, settings_json) VALUES (?, ?) "
                "ON CONFLICT(user_id) DO UPDATE SET settings_json = excluded.settings_json",
                (user_id, json.dumps(current)),
            )

    def get_settings(self, user_id: str) -> dict[str, object]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT settings_json FROM user_settings WHERE user_id = ?", (user_id,)
            ).fetchone()
        if row is None:
            return {}
        try:
            payload = json.loads(row["settings_json"])
        except (TypeError, json.JSONDecodeError):
            return {}
        return payload if isinstance(payload, dict) else {}
