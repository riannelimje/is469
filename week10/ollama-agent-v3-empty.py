# for mac in terminal
# python3 -m venv venv
# source venv/bin/activate
#
# for windows
# python -m venv venv
# .\venv\Scripts\activate
#
# then install dependencies:
# pip install mcp langchain-ollama langgraph langchain-core fastmcp
# ollama serve
# ollama serve --model llama3.2 (or whichever local LLM you have stored on your machine )

import asyncio
import datetime
from typing import Annotated, TypedDict, List

# Consolidate all imports
from mcp import ClientSession, types
from mcp.client.streamable_http import streamablehttp_client
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool, tool
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode


# --------------------------------------------------------------------------
# 1. MCP TO FETCH MS LEARN INFO 
# --------------------------------------------------------------------------
async def ask_mcp(question: str) -> str:
    SERVER_URL = "https://learn.microsoft.com/api/mcp"
    mcp_results = []
    
    async with streamablehttp_client(SERVER_URL) as (read_stream, write_stream, _):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            result = await session.call_tool("microsoft_docs_search", arguments={"query": question})
            
            if hasattr(result, "content") and result.content:
                for content in result.content:
                    if isinstance(content, types.TextContent):
                        # Store the raw text result
                        mcp_results.append(content.text)
            
            return "\n---\n".join(mcp_results) if mcp_results else "No doc found."


# --------------------------------------------------------------------------
# 2. TOOLS CREATED BY DECORATORS AND SIMPLE PYTHON FCNS
# --------------------------------------------------------------------------
@tool
def math_eval(expression: str) -> str:
    """Evaluates a mathematical expression in pure python"""
    try:
        return str(eval(expression, {"__builtins__": {}}))
    except Exception as e:
        return f"Error: {e}"
    
@tool
def current_datetime(format: str = "%Y-%m-%d") -> str:
    """Returns the current date and time in the specified format."""
    return datetime.datetime.now().strftime(format)



tools = [math_eval, current_datetime]

# --------------------------------------------------------------------------
# 3. LangGraph Agent  (LLM + Tools + Memory)
# --------------------------------------------------------------------------
llm = ChatOllama(model="llama3.2:1b", temperature=0).bind_tools(tools)

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages] # memory - remember the history of messages
    next_step: str

memory = InMemorySaver()

def call_llm(state: AgentState):
    # Minimal function to invoke the LLM
    return {"messages": [llm.invoke(state["messages"])]}


class CustomToolNode(ToolNode):
    """ToolNode class with custom tool lookup and output storing."""
    def __init__(self, tools: List[BaseTool]):
        super().__init__(tools)
        self.tool_list = tools

    def invoke(self, state, config=None):
        msgs = []
        tool_calls = state["messages"][-1].tool_calls

        for call in tool_calls:
            name, args = call["name"], call.get("args", {})
            tool = next((t for t in self.tool_list if t.name == name), None)
            
            if not tool:
                result = f"[Tool error: Tool '{name}' not found]"
            else:
                try:
                    result = tool.invoke(args)
                except Exception as e:
                    result = f"[Tool error: {e}]"

            msgs.append(
                ToolMessage(content=str(result), name=name, tool_call_id=call.get("id", "no-id"))
            )
        return {"messages": msgs}

tool_node = CustomToolNode(tools)

def router(state: AgentState):
    if getattr(state["messages"][-1], "tool_calls", None): # decide on which tools to call
        return {"next_step": "tool"}
    return {"next_step": "end"}

# Build the langgraph
workflow = StateGraph(AgentState)
workflow.add_node("llm", call_llm)
workflow.add_node("tool", tool_node)
workflow.add_node("router", router)
workflow.set_entry_point("llm")
workflow.add_edge("llm", "router")
workflow.add_conditional_edges("router", lambda s: s["next_step"], {"tool": "tool", "end": END})
workflow.add_edge("tool", "llm")

app = workflow.compile(checkpointer=memory)

# --------------------------------------------------------------------------
# 4. INTERACT WITH THE AGENT  
# --------------------------------------------------------------------------
async def main():
    mcp_output = await ask_mcp("which is better word document or onenote?")

    THREAD_ID = "user-123"
    config = {"configurable": {"thread_id": THREAD_ID}}

    # Q1: Tool Call 
    user_input_1 = "what is the current date and time and caluculate 1000-6576?"
    resp1 = app.invoke({"messages": [HumanMessage(content=user_input_1)]}, config=config)
    print(f"\nUser 1: {user_input_1}\nAgent 1: {resp1['messages'][-1].content}")

    # Q2: Memory Recall
    user_input_2 = "what did I ask before? double check the results"
    resp2 = app.invoke({"messages": [HumanMessage(content=user_input_2)]}, config=config)
    print(f"\nUser 2: {user_input_2}\nAgent 2: {resp2['messages'][-1].content}")

    # Q3: Using MCP result
    user_question_3 = "why would you one note is better?"
    injected_message = (
        f"{user_question_3}\n\n"
        f"--- MCP result ---\n"
        f"{mcp_output}"
    )
    
    resp3 = app.invoke({"messages": [HumanMessage(content=injected_message)]}, config=config)
    print(f"\nUser 3: {user_question_3}\nAgent 3: {resp3['messages'][-1].content}")


if __name__ == "__main__":
    asyncio.run(main())