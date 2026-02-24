from langchain_mcp_adapters.client import MultiServerMCPClient

# ExaAI: web search, company research (market intel); optional deep_researcher for other use cases
EXAMPLE_MCP_ENDPOINT = "https://mcp.exa.ai/mcp"
EXAMPLE_MCP_MARKET_TOOLS = "https://mcp.exa.ai/mcp?tools=web_search_exa,company_research_exa"
EXAMPLE_MCP_WITH_DEEP_RESEARCH = (
    "https://mcp.exa.ai/mcp?tools=web_search_exa,get_code_context_exa,company_research_exa,deep_researcher_start,deep_researcher_check"
)
PISTOM_MCP_ENDPOINT = "https://pistom.fastmcp.app/mcp"


def get_streamable_http_mcp_client(include_deep_research: bool = False) -> MultiServerMCPClient:
    """
    Returns an MCP Client for AgentCore Gateway compatible with LangGraph.
    Default: Exa market tools only (web_search_exa, company_research_exa). Set include_deep_research=True for deep_researcher_*.
    """
    url = EXAMPLE_MCP_WITH_DEEP_RESEARCH if include_deep_research else EXAMPLE_MCP_MARKET_TOOLS
    return MultiServerMCPClient(
        {
            "example_endpoint": {
                "transport": "streamable_http",
                "url": url,
            }
            # "pistom_endpoint": {
            #     "transport": "streamable_http",
            #     "url": PISTOM_MCP_ENDPOINT,
            # }
            # "piston": {
            #     "command": "uv",
            #     "args": [
            #         "tool",
            #         "run",
            #         "--from",
            #         "git+https://github.com/alvii147/piston-mcp.git@main",
            #         "piston_mcp"
            #     ]
            #     }
        }
    )