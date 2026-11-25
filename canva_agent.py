import asyncio
import json
import os
import sys
import urllib.request
from pathlib import Path
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage # input
from langchain_core.tools import tool # tools
from langchain_google_genai import ChatGoogleGenerativeAI # llm
from langchain_mcp_adapters.client import MultiServerMCPClient # mcp client
from langchain_mcp_adapters.tools import load_mcp_tools # mcp tools
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph.message import add_messages
from typing import Annotated, TypedDict


load_dotenv()
MODEL = os.getenv("MODEL")
DOWNLOADS = Path(__file__).with_name("downloads")

SYSTEM_PROMPT = """You create a design in Canva, then export it as a PNG.

1. Call create-design with the user's brief.
2. The reply has job_id, continuation_token, and polling_policy.wait_seconds.
3. Call wait for that many seconds.
4. Call get-create-design-async-job with the same job_id and the newest continuation_token.
5. Repeat steps 3 and 4 until the reply includes design.id.
6. Call export-design with that design_id and format {"type": "png"}.

Pass a short user_intent on every Canva tool call. Then stop."""

@tool
async def wait(seconds: int) :
    """Wait for a given number of seconds."""
    await asyncio.sleep(seconds)
    return f"Waited for {seconds} seconds."

class State(TypedDict):
    messages: Annotated[list, add_messages]

def build_agent(tools, api_key):
    model = ChatGoogleGenerativeAI(model=MODEL, api_key=api_key)
    model = model.bind_tools(tools)

    async def agent(state: State) :
        reply = await model.ainvoke([SystemMessage(SYSTEM_PROMPT), *state["messages"]])
        return {"messages": [reply]}
    
    graph = StateGraph(State)
    graph.add_node("agent", agent)
    graph.add_node("tools", ToolNode(tools))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")
    output = graph.compile()
    print(output.get_graph().draw_ascii())
    return output

async def main(prompt):
    api_key = os.getenv("GOOGLE_API_KEY")
    canva = MultiServerMCPClient({
        "canva": {
            "transport": "stdio",
            "command": "npx",
            "args": ["-y", "mcp-remote@latest", "https://mcp.canva.com/mcp"],
        }
    }
        )
    
    async with canva.session("canva") as session:
        canva_tools = await load_mcp_tools(session)
        tools_dict = {tool.name: tool for tool in canva_tools}
        print(f"Complete Canva Tools available: {tools_dict.keys()}")
        tools = [wait, tools_dict["create-design"], tools_dict["get-create-design-async-job"], tools_dict["export-design"]]

        app = build_agent(tools, api_key)

        result = await app.ainvoke({"messages": [HumanMessage(prompt)]})

        donwload_url = None

        for message in result["messages"]:
            if not isinstance(message, ToolMessage) or message.name != "export-design":
                continue

            text = message.content[0]["text"] if isinstance(message.content, list) else message.content
            donwload_url = json.loads(text)["job"]["urls"][0]
            urllib.request.urlretrieve(donwload_url, DOWNLOADS / "design.png")
            print(f"Design downloaded to {DOWNLOADS / 'design.png'}")
    
        path = DOWNLOADS / "design.png"
        urllib.request.urlretrieve(donwload_url, path)
        print(f"Design downloaded to {path}")
       
    

if __name__ == "__main__":
    DOWNLOADS.mkdir( exist_ok=True)
    asyncio.run(main("Create an Instagram post for Manifold AI Learning with tagline of 'Systems Ship, Demos Don't', create in Dark Industrial Look"))