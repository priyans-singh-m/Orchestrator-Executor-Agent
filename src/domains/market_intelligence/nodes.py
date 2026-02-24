"""Market Intelligence domain nodes."""

import logging
from typing import Any, Callable

from langchain_core.messages import SystemMessage, HumanMessage

from domains.market_intelligence.state import MarketIntelligenceState
from utils.observability import timer

logger = logging.getLogger(__name__)


SCOUT_SYSTEM = """You are a market research assistant. For the given research subject, generate a short paragraph of plausible market data (2-4 sentences). Include: typical price range, availability, and one or two competitive or deal-related points. Be concise and factual in tone. Output only the paragraph, no headings."""

STRATEGIST_SYSTEM = """You are a strategy analyst. Given the aggregated market data below, write a clear, readable strategy report for the user. Structure it with:
1. A brief summary of what was researched.
2. Key findings (pricing, availability, deals) in plain language.
3. A concrete pricing strategy recommendation to be the most competitive seller.
Use short paragraphs and bullet points where helpful. Write in a professional but accessible tone. Output the report only, no meta-commentary."""


def create_scout_node(llm: Any, data_fetcher: Any = None):
    """Create scout node: use data_fetcher if provided (async, uses MCP ainvoke), else LLM generates plausible market data.
    data_fetcher(subject) must be async and return a dict with 'summary' when provided."""
    @timer("scout_node")
    async def scout_node(state: MarketIntelligenceState) -> dict:
        tickers = state.get("tickers") or []
        subject = tickers[0] if tickers else "UNKNOWN"
        logger.info("[Node] scout_node called for subject=%s (data_fetcher=%s)", subject, data_fetcher is not None)
        if data_fetcher:
            try:
                data = await data_fetcher(subject)
                return {"raw_data": [{"ticker": subject, "summary": data.get("summary", str(data))}]}
            except Exception as e:
                logger.warning("[Node] scout_node data_fetcher failed: %s", e)
        # LLM fallback when no data_fetcher or it failed
        try:
            response = llm.invoke([
                SystemMessage(content=SCOUT_SYSTEM),
                HumanMessage(content=f"Research subject: {subject}"),
            ])
            summary = (response.content or "").strip() or "No data available."
            return {"raw_data": [{"ticker": subject, "summary": summary}]}
        except Exception:
            return {"raw_data": [{"ticker": subject, "summary": "Data unavailable for this subject."}]}
    return scout_node


@timer("synthesizer_node")
def synthesizer_node(state: MarketIntelligenceState) -> dict:
    """Merge raw_data into a single aggregated_metrics view using actual content from each item."""
    raw = state.get("raw_data") or []
    logger.info("[Node] synthesizer_node called (items=%d)", len(raw))
    lines = []
    for d in raw:
        ticker = d.get("ticker", "?")
        content = d.get("summary") or d.get("data") or d.get("content", "No data.")
        lines.append(f"## {ticker}\n{content}")
    aggregated = "\n\n".join(lines) if lines else "No data."
    return {"aggregated_metrics": aggregated}


def create_strategist_node(llm: Any):
    """Create strategist node: LLM turns aggregated_metrics into a readable strategy report."""
    def strategist_node(state: MarketIntelligenceState) -> dict:
        metrics = state.get("aggregated_metrics") or ""
        logger.info("[Node] strategist_node called (aggregated_metrics len=%d)", len(metrics))
        if not metrics.strip():
            return {"final_summary": "No market data was available to generate a report."}
        try:
            response = llm.invoke([
                SystemMessage(content=STRATEGIST_SYSTEM),
                HumanMessage(content=metrics),
            ])
            report = (response.content or "").strip() or metrics
            return {"final_summary": report}
        except Exception:
            return {"final_summary": f"Strategy report (based on aggregated data):\n{metrics}"}
    return strategist_node
