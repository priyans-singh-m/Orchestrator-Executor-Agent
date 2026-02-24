"""Market Intelligence workflow: fan-out scouts, fan-in to synthesizer and strategist."""

import logging
from typing import Any

from langgraph.graph import START, END, StateGraph
from langgraph.types import Send

from domains.market_intelligence.state import MarketIntelligenceState
from domains.market_intelligence.nodes import (
    create_scout_node,
    synthesizer_node,
    create_strategist_node,
)

logger = logging.getLogger(__name__)


def _dispatch_scouts(state: MarketIntelligenceState) -> list[Send]:
    """Fan-out: one Send per ticker to scout_node; if no tickers, send to synthesizer."""
    tickers = state.get("tickers") or []
    if not tickers:
        logger.info("[Subgraph] dispatch: no tickers -> synthesizer_node")
        return [Send("synthesizer_node", state)]
    logger.info("[Subgraph] dispatch: fan-out to scout_node x%d for tickers=%s", len(tickers), tickers)
    return [
        Send("scout_node", {**state, "tickers": [t]})
        for t in tickers
    ]


def get_market_intelligence_graph(llm: Any, data_fetcher: Any = None):
    """Build and return the compiled Market Intelligence subgraph. Requires llm for scout and strategist.
    data_fetcher: optional sync (subject -> dict with 'summary'); e.g. Exa deep research."""
    builder = StateGraph(MarketIntelligenceState)
    builder.add_node("scout_node", create_scout_node(llm, data_fetcher=data_fetcher))
    builder.add_node("synthesizer_node", synthesizer_node)
    builder.add_node("strategist_node", create_strategist_node(llm))

    builder.add_conditional_edges(START, _dispatch_scouts)
    builder.add_edge("scout_node", "synthesizer_node")
    builder.add_edge("synthesizer_node", "strategist_node")
    builder.add_edge("strategist_node", END)

    return builder.compile()
