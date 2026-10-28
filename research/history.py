from __future__ import annotations

import json
import os
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

from filelock import FileLock


HISTORY_SCHEMA_VERSION = 1


class ResearchHistoryStore:
    def __init__(self, account_dir: Path):
        self.path = Path(account_dir) / "research_history" / "history.json"
        self.lock = FileLock(str(self.path) + ".lock", timeout=15)

    def _read_unlocked(self) -> list[dict]:
        if not self.path.exists():
            return []
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            if payload.get("schema_version") != HISTORY_SCHEMA_VERSION or not isinstance(payload.get("records"), list):
                raise ValueError("Unsupported history schema.")
            return payload["records"]
        except Exception as exc:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            quarantine = self.path.with_name(f"history.corrupt-{stamp}.json")
            try:
                os.replace(self.path, quarantine)
            except OSError:
                pass
            raise ValueError(f"Research history was invalid and preserved as {quarantine.name}.") from exc

    def _write_unlocked(self, records: list[dict]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        content = json.dumps({"schema_version": HISTORY_SCHEMA_VERSION, "records": records}, indent=2, ensure_ascii=False)
        descriptor, temp_name = tempfile.mkstemp(prefix="history-", suffix=".tmp", dir=self.path.parent)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as target:
                target.write(content)
                target.flush()
                os.fsync(target.fileno())
            os.replace(temp_name, self.path)
        finally:
            try:
                os.unlink(temp_name)
            except FileNotFoundError:
                pass

    def add_result(self, result) -> dict:
        record = {
            "research_id": str(uuid.uuid4()),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "question": result.question,
            "report_markdown": result.report_markdown,
            "sources": [source.__dict__ if hasattr(source, "__dict__") else {
                "source_id": source.source_id, "kind": source.kind, "title": source.title,
                "text": source.text, "url": source.url, "published_at": source.published_at,
                "retrieved_at": source.retrieved_at, "metadata": source.metadata,
            } for source in result.sources],
            "findings": [{
                "role": item.role, "task_id": item.task_id, "assignment": item.assignment,
                "summary": item.summary, "evidence_ids": item.evidence_ids,
                "uncertainties": item.uncertainties, "error": item.error,
            } for item in result.findings],
            "trace": {
                "events": result.trace.events,
                "assignments": [{"task_id": task.task_id, "question": task.question} for task in result.trace.assignments],
                "review_rounds": [{
                    "issues": [{"issue_id": issue.issue_id, "concern": issue.concern,
                                "evidence_ids": issue.evidence_ids, "severity": issue.severity}
                               for issue in review.issues],
                    "summary": review.summary, "error": review.error,
                } for review in result.trace.review_rounds],
                "issue_dispositions": result.trace.issue_dispositions,
                "unresolved_issues": result.trace.unresolved_issues,
            },
        }
        with self.lock:
            records = self._read_unlocked()
            records.append(record)
            self._write_unlocked(records)
        return record

    def list_all(self) -> list[dict]:
        with self.lock:
            return sorted(self._read_unlocked(), key=lambda item: item.get("created_at", ""), reverse=True)

    def delete(self, research_id: str) -> bool:
        with self.lock:
            records = self._read_unlocked()
            remaining = [record for record in records if record.get("research_id") != research_id]
            if len(records) == len(remaining):
                return False
            self._write_unlocked(remaining)
            return True

    def clear(self) -> int:
        with self.lock:
            records = self._read_unlocked()
            self._write_unlocked([])
            return len(records)
