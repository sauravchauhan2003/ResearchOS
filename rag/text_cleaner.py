from __future__ import annotations

import re
import unicodedata


def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).replace("\x00", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(re.sub(r"[\t\u00a0 ]+", " ", line).strip() for line in text.split("\n"))
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
