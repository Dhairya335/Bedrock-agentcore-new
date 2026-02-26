import os
import re  ## ADDED: For stripping <thinking> tags
from strands import Agent, tool
from strands_tools.code_interpreter import AgentCoreCodeInterpreter
from bedrock_agentcore.memory.integrations.strands.config import AgentCoreMemoryConfig, RetrievalConfig
from bedrock_agentcore.memory.integrations.strands.session_manager import AgentCoreMemorySessionManager
from bedrock_agentcore.runtime import BedrockAgentCoreApp
from mcp_client.client import get_streamable_http_mcp_client
from model.load import load_model

app = BedrockAgentCoreApp()
log = app.logger

MEMORY_ID = os.getenv("BEDROCK_AGENTCORE_MEMORY_ID")
REGION = os.getenv("AWS_REGION")

# Import AgentCore Gateway as Streamable HTTP MCP Client
mcp_client = get_streamable_http_mcp_client()

@tool
def add_numbers(a: int, b: int) -> int:
    """Return the sum of two numbers"""
    return a+b

@app.entrypoint
async def invoke(payload, context):
    session_id = getattr(context, 'session_id', 'default')
    user_id = payload.get("user_id") or 'default-user'
    
    # Configure memory
    session_manager = None
    if MEMORY_ID:
        session_manager = AgentCoreMemorySessionManager(
            AgentCoreMemoryConfig(
                memory_id=MEMORY_ID,
                session_id=session_id,
                actor_id=user_id,
                retrieval_config={
                    f"/facts/{user_id}/": RetrievalConfig(top_k=10, relevance_score=0.4),
                    f"/preferences/{user_id}/": RetrievalConfig(top_k=5, relevance_score=0.5),
                    f"/summaries/{user_id}/{session_id}/": RetrievalConfig(top_k=5, relevance_score=0.4),
                    f"/episodes/{user_id}/{session_id}/": RetrievalConfig(top_k=5, relevance_score=0.4),
                }
            ),
            REGION
        )

    # Create code interpreter
    code_interpreter = AgentCoreCodeInterpreter(
        region=REGION,
        session_name=session_id,
        auto_create=True,
        persist_sessions=True
    )

    with mcp_client as client:
        tools = client.list_tools_sync()

        agent = Agent(
            model=load_model(),
            session_manager=session_manager,
            system_prompt="""
                You are a helpful assistant with code execution capabilities. 
                Respond directly to the user. Do not include your internal thought process.
            """,
            tools=[code_interpreter.code_interpreter, add_numbers] + tools
        )

        # Execute and format response
        stream = agent.stream_async(payload.get("prompt"))

        ## CHANGED: State management to filter out thinking blocks
        is_thinking = False

        async for event in stream:
            # Check for text data in the event
            if "data" in event and isinstance(event["data"], str):
                chunk = event["data"]

                ## ADDED: Logic to suppress everything between <thinking> and </thinking>
                if "<thinking" in chunk:
                    is_thinking = True
                    continue
                if "</thinking>" in chunk:
                    is_thinking = False
                    continue
                
                # Only yield if we are not inside a thinking block
                if not is_thinking:
                    # Clean up any residual whitespace or formatting if necessary
                    yield chunk

def format_response(result) -> str:
    """Extract code from metrics and format with LLM response."""
    parts = []
    try:
        tool_metrics = result.metrics.tool_metrics.get('code_interpreter')
        if tool_metrics and hasattr(tool_metrics, 'tool'):
            action = tool_metrics.tool['input']['code_interpreter_input']['action']
            if 'code' in action:
                parts.append(f"## Executed Code:\n```{action.get('language', 'python')}\n{action['code']}\n```\n---\n")
    except (AttributeError, KeyError):
        pass

    parts.append(f"## 📊 Result:\n{str(result)}")
    return "\n".join(parts)

if __name__ == "__main__":
    app.run()