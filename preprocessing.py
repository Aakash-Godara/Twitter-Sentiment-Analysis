"""
Shared tweet-cleaning function used by the notebook, predict.py, and the
Streamlit demo, so training-time and inference-time cleaning can never drift
apart.
"""
from __future__ import annotations

import html
import re

import contractions

URL_RE = re.compile(r"(https?://\S+|www\.\S+)")
MENTION_RE = re.compile(r"@\w+")
NONALPHA_RE = re.compile(r"[^a-zA-Z\s]")
MULTISPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    text = html.unescape(text)  # &amp; &quot; &lt; etc. show up raw in Sentiment140
    text = text.lower()
    text = URL_RE.sub(" ", text)
    text = MENTION_RE.sub(" ", text)
    try:
        text = contractions.fix(text)
    except Exception:
        pass
    text = NONALPHA_RE.sub(" ", text)
    text = MULTISPACE_RE.sub(" ", text).strip()
    return text
