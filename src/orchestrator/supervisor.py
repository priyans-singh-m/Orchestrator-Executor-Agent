"""Supervisor: router LLM that detects intent and sets next_node for routing."""

import logging

from langchain_core.messages import SystemMessage, HumanMessage

from orchestrator.state import GlobalState

logger = logging.getLogger(__name__)


ROUTER_SYSTEM = """You are a router. Classify the user's latest message into exactly one category:
- "market_intelligence" if the user asks for competitive analysis, market trends, competitor comparison, or stock/market data.
- "main_agent" for anything else (general chat, coding, other questions).

Reply with only one of these two words: market_intelligence or main_agent."""


def create_supervisor_node(llm):
    """Create the supervisor node that uses the LLM to route."""

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
            logger.info("[Node] supervisor -> main_agent (no query)")
            return {"next_node": "main_agent"}
        response = llm.invoke([
            SystemMessage(content=ROUTER_SYSTEM),
            HumanMessage(content=query),
        ])
        text = (response.content or "").strip().lower()
        if "market_intelligence" in text:
            logger.info("[Node] supervisor -> market_intelligence")
            return {"next_node": "market_intelligence"}
        logger.info("[Node] supervisor -> main_agent")
        return {"next_node": "main_agent"}

    return supervisor_node
