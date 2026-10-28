from __future__ import annotations

import hashlib
import hmac
import os
import pickle
import secrets
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from filelock import FileLock

from memory.models import MEMORY_SCHEMA_VERSION, MemoryRecord


MAGIC = b"RESEARCHOS-PICKLE-V1\n"


class MemoryIntegrityError(Exception):
    """Raised when the memory file was not written by this application store."""


class PickleStore:
    """HMAC-authenticated pickle storage for application-generated records only."""

    def __init__(self, path: Path, signing_key_path: Path, signing_key: bytes | None = None):
        self.path = Path(path)
        self.signing_key_path = Path(signing_key_path)
        self.lock = FileLock(str(self.path) + ".lock", timeout=15)
        self._provided_key = signing_key

    def _key(self) -> bytes:
        if self._provided_key is not None:
            if len(self._provided_key) < 32:
                raise ValueError("Memory signing keys must be at least 32 bytes.")
            return self._provided_key
        self.signing_key_path.parent.mkdir(parents=True, exist_ok=True)
        key_lock = FileLock(str(self.signing_key_path) + ".lock", timeout=15)
        with key_lock:
            try:
                return self.signing_key_path.read_bytes()
            except FileNotFoundError:
                key = secrets.token_bytes(32)
                try:
                    descriptor = os.open(self.signing_key_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
                except FileExistsError:
                    return self.signing_key_path.read_bytes()
                with os.fdopen(descriptor, "wb") as target:
                    target.write(key)
                    target.flush()
                    os.fsync(target.fileno())
                return key

    def _encode(self, payload: dict[str, Any]) -> bytes:
        serialized = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
        signature = hmac.new(self._key(), serialized, hashlib.sha256).hexdigest().encode("ascii")
        return MAGIC + signature + b"\n" + serialized

    def _decode(self, raw: bytes) -> dict[str, Any]:
        if not raw.startswith(MAGIC):
            raise MemoryIntegrityError("Memory file is not a ResearchOS memory file.")
        signature, separator, serialized = raw[len(MAGIC):].partition(b"\n")
        if not separator:
            raise MemoryIntegrityError("Memory file signature is missing.")
        expected = hmac.new(self._key(), serialized, hashlib.sha256).hexdigest().encode("ascii")
        if not hmac.compare_digest(signature, expected):
            raise MemoryIntegrityError("Memory file failed integrity verification.")
        # Authenticate before unpickling: malformed or replaced files cannot run pickle payloads.
        payload = pickle.loads(serialized)
        if not isinstance(payload, dict) or payload.get("schema_version") != MEMORY_SCHEMA_VERSION:
            raise MemoryIntegrityError("Unsupported memory schema version.")
        records = payload.get("records")
        if not isinstance(records, list):
            raise MemoryIntegrityError("Memory record list is invalid.")
        return payload

    def load_unlocked(self) -> tuple[dict[str, Any], str | None]:
        if not self.path.exists():
            return {"schema_version": MEMORY_SCHEMA_VERSION, "records": []}, None
        try:
            return self._decode(self.path.read_bytes()), None
        except Exception as exc:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            quarantine = self.path.with_name(f"{self.path.name}.corrupt-{stamp}")
            try:
                os.replace(self.path, quarantine)
            except OSError:
                quarantine = self.path
            return {"schema_version": MEMORY_SCHEMA_VERSION, "records": []}, (
                f"Memory data could not be trusted ({type(exc).__name__}); preserved at {quarantine.name}."
            )

    def write_unlocked(self, payload: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = self._encode(payload)
        descriptor, temporary_name = tempfile.mkstemp(prefix="memory-", suffix=".tmp", dir=self.path.parent)
        try:
            with os.fdopen(descriptor, "wb") as target:
                target.write(data)
                target.flush()
                os.fsync(target.fileno())
            os.replace(temporary_name, self.path)
        finally:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass

    def read(self) -> tuple[list[MemoryRecord], str | None]:
        with self.lock:
            payload, warning = self.load_unlocked()
            records = [MemoryRecord.model_validate(item) for item in payload["records"]]
            return records, warning
