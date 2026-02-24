"""Exa MCP tools for Market Intelligence scout: web_search_exa (primary), company_research_exa (optional for company names)."""

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


def _find_tool(tools: list[Any], name_substring: str) -> Any | None:
    """Return the first tool whose name contains name_substring."""
    for t in tools:
        n = getattr(t, "name", None) or ""
        if name_substring in n:
            return t
    return None


def _to_summary_text(value: Any, max_items: int = 15) -> str:
    """Safely turn tool result (list, dict, or str) into a single summary string."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts = []
        for i, item in enumerate(value):
            if i >= max_items:
                break
            if isinstance(item, str):
                parts.append(item.strip())
            elif isinstance(item, dict):
                # Common Exa result shape: title, url, text/content
                text = item.get("text") or item.get("content") or item.get("summary") or item.get("title") or str(item)
                parts.append(str(text).strip())
            else:
                parts.append(str(item))
        return "\n\n".join(p for p in parts if p)
    if isinstance(value, dict):
        raw = (
            value.get("summary")
            or value.get("text")
            or value.get("content")
            or value.get("results")
        )
        if raw is not None:
            return _to_summary_text(raw)
        return str(value)
    return str(value)


async def _research_via_web_search(tool: Any, research_query: str) -> dict | None:
    """Use web_search_exa with correct arg 'query'. Handles list or dict result safely."""
    try:
        logger.info("[MCP tool] Calling web_search_exa with query: %s", research_query[:80])
        result = await tool.ainvoke({"query": research_query})
        if result is None:
            return None
        summary = _to_summary_text(result)
        if not summary:
            return None
        return {"summary": summary}
    except Exception as e:
        logger.warning("[MCP tool] web_search_exa failed: %s", e)
        return None


def _looks_like_company_name(subject: str) -> bool:
    """Heuristic: subject is a short company name (1–2 words, no digits, not a product model)."""
    s = (subject or "").strip()
    if not s or len(s) > 50:
        return False
    if re.search(r"\d", s):
        return False
    words = s.split()
    return 1 <= len(words) <= 3


async def _research_via_company(tool: Any, company_name: str) -> dict | None:
    """Use company_research_exa with correct arg 'companyName'. Handles list or dict result safely."""
    try:
        logger.info("[MCP tool] Calling company_research_exa for companyName: %s", company_name)
        result = await tool.ainvoke({"companyName": company_name})
        if result is None:
            return None
        summary = _to_summary_text(result)
        if not summary:
            return None
        return {"summary": summary}
    except Exception as e:
        logger.warning("[MCP tool] company_research_exa failed: %s", e)
        return None


def create_exa_deep_research_fetcher(tools: list[Any]) -> Any:
    """
    Build an async data_fetcher for Market Intelligence scout.
    Uses web_search_exa (primary) and optionally company_research_exa when subject looks like a company name.
    Correct args: web_search_exa(query=...), company_research_exa(companyName=...).
    Returns async f(subject: str) -> dict with key "summary". Safe for list or dict results.
    """
    web_search = _find_tool(tools, "web_search_exa")
    company_research = _find_tool(tools, "company_research_exa")

    async def fetcher(subject: str) -> dict:
        research_query = (
            f"Market and pricing research: current pricing, availability, open box or deal options, "
            f"and competitive landscape for: {subject}. Provide a concise factual summary."
        )
        logger.info("[Scout data_fetcher] Researching subject: %s", subject)

        # 1) Primary: web_search_exa (correct arg: query)
        if web_search:
            out = await _research_via_web_search(web_search, research_query)
            if out:
                return out

        # 2) Optional: company_research_exa only when subject looks like a company name (correct arg: companyName)
        if company_research and _looks_like_company_name(subject):
            out = await _research_via_company(company_research, subject)
            if out:
                return out

        return {"summary": "No Exa research tool available or all tools failed."}

    return fetcher
