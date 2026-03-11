from __future__ import annotations

import json
import re
from typing import Dict, List

import requests
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
from langchain.tools import tool


def _clean_text(html: str, max_chars: int = 40_000) -> str:
    """Convert HTML to readable text and truncate for context safety."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    text = soup.get_text("\n")
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    if len(text) > max_chars:
        text = text[:max_chars] + "\n\n[TRUNCATED]"
    return text


@tool
def internet_search(query: str) -> str:
    """Search the web and return JSON string of ranked results.

    Returns (JSON string):
      [ {"title": str, "url": str, "snippet": str}, ... ]

    Note: Uses DuckDuckGo via duckduckgo-search (no API key required).
    If your course already provides internet_search, keep the same signature
    but ensure the returned format includes title/url/snippet.
    """
    results: List[Dict[str, str]] = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=10):
            url = r.get("href") or r.get("url")
            if not url:
                continue
            results.append(
                {
                    "title": (r.get("title") or "").strip(),
                    "url": url.strip(),
                    "snippet": (r.get("body") or r.get("snippet") or "").strip(),
                }
            )

    return json.dumps(results, ensure_ascii=False)


@tool
def fetch_url(url: str) -> str:
    """Fetch page content from a URL and return readable text (best-effort)."""
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; assignment-2-agent/1.0)",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }
    resp = requests.get(url, headers=headers, timeout=15)
    resp.raise_for_status()

    ctype = (resp.headers.get("Content-Type") or "").lower()
    if "text/html" in ctype:
        return _clean_text(resp.text)

    text = resp.text
    if len(text) > 40_000:
        text = text[:40_000] + "\n\n[TRUNCATED]"
    return text
