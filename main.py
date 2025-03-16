import os
import numpy as np
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

# Initialize the LLM
# Initialize LLM with GPT-4 and verbose output
llm = ChatOpenAI(model="gpt-4", temperature=0)

# Type definitions
State = Dict[str, Any]
T = TypeVar("T")

def collect_patents(query: str, score_threshold: int) -> Dict[str, Any]:
    """Collect patents directly using the firecrawl tool without LLM."""
    return firecrawl_tool.func(query, score_threshold)

def get_top_percentile_patents(patents: list, percentile: float = 90) -> list:
    """Get patents above the specified percentile based on score."""
    if not patents:
        return []
    
    scores = np.array([p['score'] for p in patents])
    score_threshold = np.percentile(scores, percentile)
    return [p for p in patents if p['score'] >= score_threshold]

def setup_llm_analysis():
    """Set up the LLM workflow for analyzing patents."""
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
            return {"messages": messages}
        
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
        """Process the LLM's function call."""
        messages = state["messages"]
        last_message = messages[-1]
        
        if not last_message.additional_kwargs.get("function_call"):
            return END
        
        function_call = last_message.additional_kwargs["function_call"]
        return {
            "tool": function_call["name"],
            "tool_input": function_call["arguments"],
            "messages": messages,
            "function_call": function_call
        }

    def process_tool_output(state: State) -> Dict[str, Any]:
        """Process the tool output."""
        messages = state.get("messages", [])
        output = state.get("output")
        function_call = state.get("function_call")
        
        if not all([messages, output, function_call]):
            return {"messages": messages}
        
        function_message = FunctionMessage(
            content=str(output),
            name=function_call["name"],
        )
        
        if function_call["name"] == "firecrawl_patent_scraper" and isinstance(output, dict):
            if "formatted_output" in output:
                print(output["formatted_output"])
            
            return {
                "messages": [*messages, function_message],
                "patents": output.get("patents", []),
                "metadata": output.get("metadata", {})
            }
        
        return {"messages": [*messages, function_message]}

    # Create and configure the workflow graph
    workflow = Graph()
    workflow.add_node("llm", call_llm)
    workflow.add_node("process_function_call", process_function_call)
    workflow.add_node("run_tool", run_tool)
    workflow.add_node("process_tool_output", process_tool_output)
    
    workflow.add_edge("llm", "process_function_call")
    workflow.add_conditional_edges(
        "process_function_call",
        lambda x: "end" if isinstance(x, str) else "run_tool",
        {"end": END, "run_tool": "run_tool"}
    )
    workflow.add_edge("run_tool", "process_tool_output")
    workflow.add_edge("process_tool_output", "llm")
    
    workflow.set_entry_point("llm")
    return workflow.compile()

def analyze_patents_with_llm(patents: list, user_interest: str, llm: ChatOpenAI, chunk_size: int = 1000, verbose_explanations: bool = False):
    """Analyze patents using LLM with sliding window memory and user's interest focus.
    
    Args:
        patents: List of patents to analyze
        user_interest: User's specific interest or focus area
        llm: LangChain ChatOpenAI instance to use for analysis
        chunk_size: Size of text chunks to process (default: 1000)
        verbose_explanations: Whether to show LLM's reasoning for each chunk (default: False)
    
    Returns:
        List of analyzed patents with relevance scores
    """
    from tools import analyze_patent_content
    # LLM is now passed as a parameter
    
    print("\n🤖 Starting Patent Analysis")
    print("=" * 30)
    print(f"Focus: {user_interest}")
    
    relevant_findings = []
    
    for i, patent in enumerate(patents, 1):
        print(f"\nAnalyzing Patent {i}/{len(patents)}: {patent['title']}")
        
        # Get patent content
        content = analyze_patent_content(patent['url'])
        if not content:
            print("Failed to retrieve patent content")
            continue
        
        # Use description section for analysis
        text = content['sections']['description']
        if not text.strip():
            print("No description section found in patent")
            continue
            
        # Split description into chunks
        chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
        
        # Initialize memory window for this patent
        memory_window = []
        chunk_findings = []
        
        print(f"Processing {len(chunks)} chunks...")
        
        for j, chunk in enumerate(chunks, 1):
            # Create prompt with sliding window memory
            memory_context = "\n".join([f"Previous Chunk Context {k+1}:\n{mem}" 
                                      for k, mem in enumerate(memory_window[-2:])])
            
            # Choose prompt based on verbose_explanations setting
            if verbose_explanations:
                prompt = f"""User's Interest: {user_interest}
                
                Previous Context:
                {memory_context}
                
                Patent Chunk {j}/{len(chunks)}:
                {chunk}
                
                First explain in ONE brief sentence why this chunk is or isn't relevant to the user's interest.
                Then on a new line, respond with EXACTLY one word: either 'yes' or 'no'.
                """
            else:
                prompt = f"""User's Interest: {user_interest}
                
                Previous Context:
                {memory_context}
                
                Patent Chunk {j}/{len(chunks)}:
                {chunk}
                
                Does this chunk contain information relevant to the user's interest?
                Respond with EXACTLY one word: either 'yes' or 'no'.
                """
            
            messages = [
                {"role": "system", "content": "You are a patent analysis expert. Your task is to determine if a chunk of text is relevant to a specific interest. Respond with EXACTLY one word: 'yes' or 'no'."},
                {"role": "user", "content": prompt}
            ]
            
            response = llm.invoke(messages)
            
            # Update memory window with current chunk
            memory_window.append(chunk)
            if len(memory_window) > 2:
                memory_window.pop(0)
            
            # Parse response based on verbose_explanations setting
            content = response.content.strip()
            if verbose_explanations:
                lines = content.split('\n')
                explanation = lines[0].strip()
                is_relevant = lines[-1].strip().lower() == 'yes'
            else:
                is_relevant = content.lower() == 'yes'
                explanation = None
            chunk_findings.append({
                'chunk_num': j,
                'is_relevant': is_relevant
            })
            # Calculate current progress
            current_relevant = sum(1 for f in chunk_findings if f['is_relevant'])
            current_ratio = current_relevant / j
            
            print(f"\n🔎 Analyzing Chunk {j}/{len(chunks)}")
            print("=" * 30)
            print(f"📄 Content Preview:")
            print(f"`{chunk[:200]}...`")
            print(f"🤖 LLM Response: {'✅ Relevant' if is_relevant else '❌ Not relevant'}")
            if verbose_explanations:
                print(f"💬 Reason: {explanation}")
            print(f"📊 Progress: {current_relevant}/{j} chunks relevant ({current_ratio:.1%})\n")
        
        # Analyze chunk findings to determine if patent is relevant
        relevant_chunks = sum(1 for f in chunk_findings if f['is_relevant'])
        relevance_ratio = relevant_chunks / len(chunk_findings)
        
        # If at least 25% of chunks are relevant, consider the patent relevant
        if relevance_ratio >= 0.25:
            relevant_findings.append({
                'patent': patent,
                'relevance_ratio': relevance_ratio,
                'relevant_chunks': relevant_chunks,
                'total_chunks': len(chunk_findings)
            })
            print(f"Patent is relevant: {relevant_chunks}/{len(chunk_findings)} chunks relevant ({relevance_ratio:.1%})")
        else:
            print(f"Patent not relevant: only {relevant_chunks}/{len(chunk_findings)} chunks relevant ({relevance_ratio:.1%})")
            
        # Clear memory window between patents
        memory_window = []
        
    # Display relevant patents
    if relevant_findings:
        print("\n🔍 Patents Matching Your Interest")
        print("=" * 30)
        
        for rf in relevant_findings:
            # Extract patent number from URL
            url = rf['patent'].get('url', '')
            patent_number = 'N/A'
            if url:
                parts = url.split('/')
                number = parts[-1].replace('.html', '')
                if 'y' in parts[-2]:  # Published application format
                    year = parts[-2].replace('y', '')
                    patent_number = f"{year}/{number}"
                else:  # Granted patent format
                    patent_number = number

            print(f"📄 [{rf['patent']['title']}]({url})")
            print(f"📊 Relevance Score: {rf['relevance_ratio']:.1%} ({rf['relevant_chunks']}/{rf['total_chunks']} chunks)")
            print(f"🔢 Patent Number: `{patent_number}`\n")
    else:
        print("\nNo patents were found to be sufficiently relevant to your interest.")
    
    return relevant_findings

def get_valid_score_threshold(score_input: str) -> int:
    """Validate and return score threshold with proper error handling."""
    try:
        score = int(score_input)
        if 0 <= score <= 1000:
            return score
        print("Score threshold must be between 0 and 1000. Using default value of 600.")
    except ValueError:
        print("Invalid score threshold. Using default value of 600.")
    return 600

def get_valid_percentile(percentile_input: str) -> float:
    """Validate and return percentile with proper error handling."""
    try:
        percentile = float(percentile_input) if percentile_input.strip() else 90
        if 0 <= percentile <= 99:
            return percentile
        print("Percentile must be between 0 and 99. Using default value of 90.")
    except ValueError:
        print("Invalid percentile. Using default value of 90.")
    return 90

def collect_and_display_patents(query: str, score_threshold: int, percentile: float) -> tuple:
    """Collect patents and display results in formatted output."""
    result = collect_patents(query, score_threshold)
    print(result["formatted_output"])
    
    top_patents = get_top_percentile_patents(result["patents"], percentile)
    if top_patents:
        print(f"\nFound {len(top_patents)} patents in the {percentile}th percentile:")
        print("\nTop Patent URLs:")
        print("-" * 50)
        for patent in top_patents:
            print(f"Score {patent['score']}: {patent['url']}")
    
    return result["patents"], top_patents

if __name__ == "__main__":
    while True:  # Main program loop
        # Phase 1: Get search parameters
        query = input("\n📝 Enter your patent search query: ")
        score_threshold = get_valid_score_threshold(input("Enter minimum score threshold (0-1000): "))
        percentile = get_valid_percentile(input("Enter percentile threshold (0-99, default 90): "))
        
        # Phase 2: Collect and display patents
        patents, top_patents = collect_and_display_patents(query, score_threshold, percentile)
        
        if not top_patents:
            retry = input("\n❌ No patents found. Would you like to try another search? (yes/no): ").lower()
            if retry != 'yes':
                break
            continue
        
        # Phase 3: Analysis loop
        while True:
            analyze = input("\n🤖 Would you like to analyze these patents with AI? (yes/no): ").lower()
            if analyze == 'yes':
                interest = input("\n🎯 What aspects of these patents interest you most? (e.g., 'natural language processing'): ")
                want_explanations = input("\n💡 Would you like detailed explanations for relevance decisions? (costs more) (yes/no): ").lower() == 'yes'
                
                # Perform analysis
                analysis_results = analyze_patents_with_llm(top_patents, interest, llm, verbose_explanations=want_explanations)
                
                # Navigation menu
                print("\n🔄 What would you like to do next?")
                print("1. Start a new patent search")
                print("2. Analyze these patents with a different focus")
                print("3. Exit")
                
                choice = input("Enter your choice (1-3): ").strip()
                if choice == '1':
                    break  # Break to outer loop for new search
                elif choice == '2':
                    continue  # Continue analysis loop
                else:
                    print("\n👋 Thank you for using the patent analysis tool!")
                    exit()
            elif analyze == 'no':
                print("\n👋 Thank you for using the patent analysis tool!")
                exit()
            else:
                print("\n⚠️ Please enter 'yes' or 'no'")
