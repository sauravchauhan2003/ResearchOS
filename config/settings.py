from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


ROOT_DIR = Path(__file__).resolve().parents[1]


@dataclass(frozen=True, slots=True)
class AppSettings:
    """Settings shared by the local ResearchOS server."""

    data_dir: Path
    groq_api_key: str | None
    gemini_api_key: str | None
    provider_timeout_seconds: float = 30.0

    @classmethod
    def from_env(cls, *, root_dir: Path = ROOT_DIR) -> "AppSettings":
        load_dotenv(root_dir / ".env", override=False)
        data_dir = Path(os.getenv("RESEARCHOS_DATA_DIR", str(root_dir / "data")))
        if not data_dir.is_absolute():
            data_dir = root_dir / data_dir
        return cls(
            data_dir=data_dir.resolve(),
            groq_api_key=os.getenv("GROQ_API_KEY") or None,
            gemini_api_key=os.getenv("GEMINI_API_KEY") or None,
        )

    @property
    def users_db(self) -> Path:
        return self.data_dir / "accounts.sqlite3"

    def user_data_dir(self, user_id: str) -> Path:
        """Return an account-scoped data path for trusted UUID account IDs."""
        import uuid

        safe_user_id = str(uuid.UUID(user_id))
        return self.data_dir / "users" / safe_user_id
