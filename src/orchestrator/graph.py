"""Orchestrator graph: supervisor, market_intelligence subgraph wrapper, main_agent."""

import json
import logging
import re
from typing import Any

from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import MemorySaver

from orchestrator.state import GlobalState
from orchestrator.supervisor import create_supervisor_node
from domains.market_intelligence import get_market_intelligence_graph
from domains.market_intelligence.exa_research import create_exa_deep_research_fetcher

logger = logging.getLogger(__name__)

# Default competitors when none are parsed from the query
DEFAULT_TICKERS = ["AAPL", "MSFT", "GOOGL"]

# Max query length / word count to treat as "ticker-style" (short symbols only)
MAX_TICKER_QUERY_LEN = 45
MAX_TICKER_QUERY_WORDS = 6

EXTRACT_SUBJECTS_SYSTEM = """You are an assistant that extracts research subjects from a user's market or competitive-intelligence query.
Given the user message, output a JSON array of 3-6 short research subjects to look up. Include:
- Product or model names (e.g. "Sony WH-1000XM5 Headphones")
- Competitor or segment terms (e.g. "big three retailers", "competitor pricing")
- Deal types if mentioned (e.g. "open box deals")
Use concise labels (a few words each). Output only the JSON array, no other text. Example: ["Sony WH-1000XM5", "competitor pricing", "open box deals"]"""

REVIEWER_SYSTEM = """You are a quality reviewer for an AI assistant. You receive the assistant's reply to the user (possibly combining multiple parts from parallel specialists).
- If the reply is complete, correct, and well-formatted, return it unchanged (output the reply exactly as-is).
- If something is missing, wrong, or unclear, return an improved version that fixes the issue while keeping the same intent.
- If the reply has multiple sections (e.g. from different specialists), keep them coherent and well-ordered; do not drop any important part.
Output only the final reply to the user, nothing else. No preamble like "Here is the corrected version"."""


def _derive_tickers(query: str) -> list[str]:
    """Extract ticker-like symbols from query only when it looks like short ticker list; else return default."""
    if not (query or query.strip()):
        return DEFAULT_TICKERS.copy()
    q = query.strip()
    # Only use word regex for short, ticker-style inputs (e.g. "AAPL MSFT GOOGL")
    if len(q) > MAX_TICKER_QUERY_LEN or len(q.split()) > MAX_TICKER_QUERY_WORDS:
        return DEFAULT_TICKERS.copy()
    words = re.findall(r"\b[A-Z]{2,5}\b", q.upper())
    if words:
        return list(dict.fromkeys(words))[:10]
    return DEFAULT_TICKERS.copy()


def _extract_research_subjects(query: str, llm: Any) -> list[str]:
    """Use LLM to extract product/competitor/deal subjects from a natural-language market query."""
    if not (query or query.strip()):
        return DEFAULT_TICKERS.copy()
    try:
        response = llm.invoke([
            SystemMessage(content=EXTRACT_SUBJECTS_SYSTEM),
            HumanMessage(content=query),
        ])
        text = (response.content or "").strip()
        # Try to parse JSON array (allow wrapped in markdown code block)
        if "```" in text:
            text = text.split("```")[1].replace("```", "").strip()
        if text.startswith("["):
            arr = json.loads(text)
            if isinstance(arr, list) and arr:
                return [str(x).strip() for x in arr if str(x).strip()][:8]
    except (json.JSONDecodeError, TypeError):
        pass
    return [query.strip()[:80]] if query.strip() else DEFAULT_TICKERS.copy()


def _get_tickers_for_query(query: str, llm: Any) -> list[str]:
    """Choose tickers/subjects: LLM extraction for long natural-language queries, heuristic for short ticker-style."""
    if not (query or query.strip()):
        return DEFAULT_TICKERS.copy()
    q = query.strip()
    if len(q) <= MAX_TICKER_QUERY_LEN and len(q.split()) <= MAX_TICKER_QUERY_WORDS:
        return _derive_tickers(query)
    return _extract_research_subjects(query, llm)


def create_market_intelligence_node(llm: Any, data_fetcher: Any = None):
    """Create the wrapper node that invokes the Market Intelligence subgraph with state mapping (async ainvoke)."""
    subgraph = get_market_intelligence_graph(llm, data_fetcher=data_fetcher)

    async def market_intelligence_node(state: GlobalState) -> dict:
        query = state.get("current_query") or ""
        tickers = _get_tickers_for_query(query, llm)
        logger.info("[Node] market_intelligence (subgraph) invoked with tickers=%s", tickers)
        mi_input: dict[str, Any] = {
            "tickers": tickers,
            "raw_data": [],
            "aggregated_metrics": "",
            "final_summary": "",
        }
        mi_output = await subgraph.ainvoke(mi_input)
        final_summary = mi_output.get("final_summary") or ""
        logger.info("[Node] market_intelligence subgraph finished, summary len=%d", len(final_summary))
        return {
            "messages": [AIMessage(content=final_summary)],
            "market_reports": [final_summary],
        }

    return market_intelligence_node


def create_reviewer_node(llm: Any):
    """Review assistant reply; when multiple parallel branches ran, combine their outputs into one reply."""
    def reviewer_node(state: GlobalState) -> dict:
        messages = state.get("messages") or []
        logger.info("[Node] reviewer called (messages=%d)", len(messages))
        if not messages:
            return {}
        # Find last human message; collect all assistant content after it (from parallel branches)
        last_human_idx = -1
        for i in range(len(messages) - 1, -1, -1):
            if isinstance(messages[i], HumanMessage):
                last_human_idx = i
                break
        parts = []
        for i in range(last_human_idx + 1, len(messages)):
            msg = messages[i]
            content = getattr(msg, "content", None) or ""
            if not isinstance(content, str):
                content = str(content)
            if content.strip():
                parts.append(content.strip())
        content = "\n\n---\n\n".join(parts) if len(parts) > 1 else (parts[0] if parts else "")
        if not content:
            return {}
        try:
            response = llm.invoke([
                SystemMessage(content=REVIEWER_SYSTEM),
                HumanMessage(content=f"Assistant's reply to check:\n\n{content}"),
            ])
            reviewed = (response.content or "").strip() or content
            return {"messages": [AIMessage(content=reviewed)]}
        except Exception:
            return {}
    return reviewer_node


def create_main_agent_node(agent_graph):
    """Create the node that invokes the main ReAct agent."""

    async def main_agent_node_async(state: GlobalState) -> dict:
        messages = state.get("messages") or []
        logger.info("[Node] main_agent (ReAct) invoked")
        result = await agent_graph.ainvoke({"messages": messages})
        new_messages = result.get("messages") or []
        # Append only the new messages (after the ones we had)
        to_append = new_messages[len(messages):] if len(new_messages) > len(messages) else [new_messages[-1]] if new_messages else []
        return {"messages": to_append}

    def main_agent_node_sync(state: GlobalState) -> dict:
        import asyncio
        return asyncio.run(main_agent_node_async(state))

    # Prefer async if agent_graph.ainvoke is used from async entrypoint
    return main_agent_node_async


def create_orchestrator_graph(llm, agent_graph, checkpointer=None, data_fetcher=None):
    """
    Build and return the compiled orchestrator graph.
    agent_graph: the compiled create_agent (ReAct) graph.
    checkpointer: e.g. SqliteSaver or MemorySaver for multi-turn.
    data_fetcher: optional sync (subject -> dict); e.g. Exa deep research for MI scout.
    """
    builder = StateGraph(GlobalState)
    builder.add_node("supervisor", create_supervisor_node(llm))
    builder.add_node("market_intelligence", create_market_intelligence_node(llm, data_fetcher=data_fetcher))
    builder.add_node("main_agent", create_main_agent_node(agent_graph))
    builder.add_node("reviewer", create_reviewer_node(llm))

    builder.add_edge(START, "supervisor")
    # Return list of node names so multiple can run in parallel (Option B: true parallel branches).
    # When adding a new domain: add_node(...), add_edge("new_domain", "reviewer"), and add to VALID_NEXT_NODES in supervisor.
    def supervisor_next(state):
        nodes = state.get("next_nodes") or ["main_agent"]
        if not isinstance(nodes, list):
            nodes = [nodes] if nodes else ["main_agent"]
        # Only allow known nodes (same set as in supervisor.VALID_NEXT_NODES)
        valid = {"market_intelligence", "main_agent"}
        out = [n for n in nodes if n in valid]
        return out if out else ["main_agent"]

    builder.add_conditional_edges("supervisor", supervisor_next)
    builder.add_edge("market_intelligence", "reviewer")
    builder.add_edge("main_agent", "reviewer")
    builder.add_edge("reviewer", END)

    return builder.compile(checkpointer=checkpointer or MemorySaver())
