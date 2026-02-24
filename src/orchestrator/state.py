"""Orchestrator global state."""

import operator
from typing import Annotated
from typing_extensions import TypedDict, NotRequired

from langgraph.graph.message import add_messages


class GlobalState(TypedDict):
    """State for the top-level orchestrator graph."""

    messages: Annotated[list, add_messages]
    current_query: str
    market_reports: Annotated[list, operator.add]
    next_nodes: NotRequired[list[str]]  # set by supervisor; 1+ nodes run in parallel
