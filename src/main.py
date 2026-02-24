"""Entrypoint: orchestrator graph with async SQLite checkpointer for multi-turn sessions."""

import logging

import aiosqlite

# Ensure orchestrator, domains, and utils loggers show INFO in console (for agentcore dev / run)
logging.basicConfig(level=logging.INFO, format="%(levelname)s [%(name)s] %(message)s")
for _logger_name in ("orchestrator", "domains", "utils"):
    logging.getLogger(_logger_name).setLevel(logging.INFO)
from langchain_core.messages import HumanMessage
from langchain.agents import create_agent
from langchain.tools import tool
from bedrock_agentcore import BedrockAgentCoreApp
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from mcp_client.client import get_streamable_http_mcp_client
from model.load import load_model
from orchestrator.graph import create_orchestrator_graph
from domains.market_intelligence.exa_research import create_exa_deep_research_fetcher

# Define a simple function tool
@tool
def add_numbers(a: int, b: int) -> int:
    """Return the sum of two numbers"""
    return a + b

# AgentCore and MCP
mcp_client = get_streamable_http_mcp_client()
app = BedrockAgentCoreApp()
llm = load_model()

# Orchestrator graph and async checkpointer built once (lazy on first request)
_orchestrator_graph = None
_checkpointer = None
_conn = None


async def _get_orchestrator_graph(tools):
    """Build orchestrator graph once with AsyncSqliteSaver; reuse. Caller must pass tools (from await mcp_client.get_tools())."""
    global _orchestrator_graph, _checkpointer, _conn
    if _orchestrator_graph is not None:
        return _orchestrator_graph
    _conn = await aiosqlite.connect("checkpoints.db")
    _checkpointer = AsyncSqliteSaver(_conn)
    agent_graph = create_agent(llm, tools=tools + [add_numbers])
    data_fetcher = create_exa_deep_research_fetcher(tools)
    _orchestrator_graph = create_orchestrator_graph(
        llm, agent_graph, checkpointer=_checkpointer, data_fetcher=data_fetcher
    )
    return _orchestrator_graph


@app.entrypoint
async def invoke(payload):
    # Payload: { "prompt": "<user input>", optional "sessionId": "<thread_id>" }
    # With checkpointer, same sessionId resumes conversation; pass only the new turn.
    prompt = payload.get("prompt", "What is Agentic AI?")
    session_id = payload.get("sessionId", "default")

    state = {
        "messages": [HumanMessage(content=prompt)],
        "current_query": prompt,
        "market_reports": [],
    }

    tools = await mcp_client.get_tools()
    graph = await _get_orchestrator_graph(tools)
    config = {"configurable": {"thread_id": session_id}}

    result = await graph.ainvoke(state, config=config)

    out_messages = result.get("messages") or []
    last_content = out_messages[-1].content if out_messages else ""
    return {"result": last_content}


if __name__ == "__main__":
    app.run()
