"""Supervisor: router LLM that detects intent and sets next_nodes for parallel routing."""

import json
import logging
import re

from langchain_core.messages import SystemMessage, HumanMessage

from orchestrator.state import GlobalState

logger = logging.getLogger(__name__)

# Node names the supervisor can route to (add new domain agents here as they are added)
VALID_NEXT_NODES = frozenset({"market_intelligence", "main_agent"})

ROUTER_SYSTEM = """You are a router. Classify the user's latest message into one or more categories.

Available categories (you may select one or several):
- "market_intelligence": use when the user asks for competitive analysis, market trends, competitor comparison, stock/market data, or company/market research.
- "main_agent": use for general chat, coding help, explanations, or any other question that does not fit the above. Also use when the user needs both specialized research AND general assistance in the same turn.

Rules:
- If the query needs only market/competitive research, reply with ["market_intelligence"].
- If the query is only general (chat, code, other), reply with ["main_agent"].
- If the query clearly needs BOTH (e.g. "compare AAPL and MSFT then write me a summary email" or "get market data and explain what it means"), reply with ["market_intelligence", "main_agent"].
- When in doubt between multiple categories, include all that apply so the user gets a complete answer.

Reply with ONLY a JSON array of category names, no other text. Examples:
["market_intelligence"]
["main_agent"]
["market_intelligence", "main_agent"]"""


def _parse_next_nodes(text: str) -> list[str]:
    """Parse JSON array from router LLM; return only valid node names, deduplicated, order preserved."""
    text = (text or "").strip().lower()
    if not text:
        return ["main_agent"]
    # Allow JSON array possibly inside markdown code block
    if "```" in text:
        match = re.search(r"\[[\s\S]*?\]", text)
        if match:
            text = match.group(0)
    if not text.startswith("["):
        # Fallback: look for known node names (with or without spaces)
        out = []
        normalized = text.replace(" ", "_")
        if "market_intelligence" in normalized:
            out.append("market_intelligence")
        if "main_agent" in normalized:
            out.append("main_agent")
        return out if out else ["main_agent"]
    try:
        arr = json.loads(text)
        if not isinstance(arr, list):
            return ["main_agent"]
        seen = set()
        result = []
        for x in arr:
            name = str(x).strip().lower().replace(" ", "_")
            if name in VALID_NEXT_NODES and name not in seen:
                seen.add(name)
                result.append(name)
        return result if result else ["main_agent"]
    except (json.JSONDecodeError, TypeError):
        return ["main_agent"]


def create_supervisor_node(llm):
    """Create the supervisor node that uses the LLM to route to one or more nodes (parallel)."""

    def supervisor_node(state: GlobalState) -> dict:
        query = state.get("current_query") or ""
        messages = state.get("messages") or []
        if messages:
            last = messages[-1]
            content = getattr(last, "content", None)
            if isinstance(content, str):
                query = content
        logger.info("[Node] supervisor (router) called, query len=%d", len(query or ""))
        if not query:
            logger.info("[Node] supervisor -> next_nodes=['main_agent'] (no query)")
            return {"next_nodes": ["main_agent"]}
        response = llm.invoke([
            SystemMessage(content=ROUTER_SYSTEM),
            HumanMessage(content=query),
        ])
        text = (response.content or "").strip()
        next_nodes = _parse_next_nodes(text)
        logger.info("[Node] supervisor -> next_nodes=%s", next_nodes)
        return {"next_nodes": next_nodes}

    return supervisor_node
