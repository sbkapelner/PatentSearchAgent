import os
from typing import Sequence, Union, Dict, Any, TypeVar, Callable
from operator import itemgetter
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, FunctionMessage
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_function
from langgraph.graph import Graph, END

from tools import firecrawl_tool

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Define our state
State = Dict[str, Any]

# Initialize the LLM
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# Type definitions
State = Dict[str, Any]
T = TypeVar("T")

# Set up tools
tools = [firecrawl_tool]
tool_functions = [convert_to_openai_function(t) for t in tools]

def get_tool_by_name(name: str) -> BaseTool:
    """Get a tool by its name."""
    for tool in tools:
        if tool.name == name:
            return tool
    raise ValueError(f"Tool {name} not found")

def run_tool(state: State) -> Dict[str, Any]:
    """Run a tool and return its output."""
    tool_name = state.get("tool")
    tool_args = state.get("tool_input")
    messages = state.get("messages", [])
    function_call = state.get("function_call")
    
    if not all([tool_name, tool_args]):
        return {"messages": messages}  # Return current state if missing data
    
    tool = get_tool_by_name(tool_name)
    result = tool.invoke(tool_args)
    
    return {
        "output": result,
        "messages": messages,
        "function_call": function_call
    }

def call_llm(state: State) -> Dict[str, Any]:
    """Call the LLM with the current state."""
    messages = state["messages"]
    response = llm.invoke(messages, functions=tool_functions)
    return {"messages": [*messages, response]}

def process_function_call(state: State) -> Union[Dict[str, Any], str]:
    """Process the LLM's function call and route to the appropriate tool."""
    messages = state["messages"]
    last_message = messages[-1]
    
    # If no function call, we're done
    if not last_message.additional_kwargs.get("function_call"):
        return END
    
    # Get the function call
    function_call = last_message.additional_kwargs["function_call"]
    return {
        "tool": function_call["name"],
        "tool_input": function_call["arguments"],
        "messages": messages,
        "function_call": function_call
    }

def process_tool_output(state: State) -> Dict[str, Any]:
    """Process the tool output and update state."""
    messages = state.get("messages", [])
    output = state.get("output")
    function_call = state.get("function_call")
    
    if not all([messages, output, function_call]):
        return {"messages": messages}  # Return current state if missing data
    
    # Create a function message with the output
    function_message = FunctionMessage(
        content=str(output),
        name=function_call["name"],
    )
    
    # Handle structured output from the patent scraper
    if function_call["name"] == "firecrawl_patent_scraper" and isinstance(output, dict):
        if "formatted_output" in output:
            print(output["formatted_output"])
        
        # Store structured data in state for potential future use
        return {
            "messages": [*messages, function_message],
            "patents": output.get("patents", []),
            "metadata": output.get("metadata", {})
        }
    
    return {"messages": [*messages, function_message]}

# Create the graph
workflow = Graph()

# Add nodes for each stage of processing
workflow.add_node("llm", call_llm)
workflow.add_node("process_function_call", process_function_call)
workflow.add_node("run_tool", run_tool)
workflow.add_node("process_tool_output", process_tool_output)

# Add edges to create the processing flow
workflow.add_edge("llm", "process_function_call")

# Add conditional edge for process_function_call
workflow.add_conditional_edges(
    "process_function_call",
    lambda x: "end" if isinstance(x, str) else "run_tool",
    {"end": END, "run_tool": "run_tool"}
)

# Complete the tool execution cycle
workflow.add_edge("run_tool", "process_tool_output")
workflow.add_edge("process_tool_output", "llm")

# Set the entry point
workflow.set_entry_point("llm")

# Compile the graph
chain = workflow.compile()

if __name__ == "__main__":
    query = input("Enter your patent search query: ")
    score_threshold = input("Enter minimum score threshold (0-1000): ")
    
    try:
        score_threshold = int(score_threshold)
        if score_threshold < 0 or score_threshold > 1000:
            print("Score threshold must be between 0 and 1000. Using default value of 600.")
            score_threshold = 600
    except ValueError:
        print("Invalid score threshold. Using default value of 600.")
        score_threshold = 600
    
    # Initialize state with both query and score threshold
    state = {"messages": [HumanMessage(content=f"Search for patents using query: {query} minimum score threshold: {score_threshold}")]}
    
    # Run the agent
    chain.invoke(state)
