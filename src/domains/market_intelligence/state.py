"""Market Intelligence domain state."""

import operator
from typing import Annotated
from typing_extensions import TypedDict


class MarketIntelligenceState(TypedDict):
    """State for the Market Intelligence subgraph."""

    tickers: list[str]
    raw_data: Annotated[list, operator.add]
    aggregated_metrics: str
    final_summary: str
