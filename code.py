import streamlit as st
from typing import TypedDict
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# Initialize LLM
llm = ChatGroq(model="qwen-2.5-32b")
llm_router = ChatGroq(model="qwen-2.5-32b")

# Define State
class State(TypedDict):
    input: str
    decision: str
    output: str
    test_result: bool
    review_result: bool

# Nodes
def review(state: State):
    """Review the code"""
    st.write("Reviewing the code")
    result = llm.invoke(state["input"])
    review_passed = True  # Simulated review result
    return {"output": result.content, "review_result": review_passed}

def generate_code(state: State):
    """Generate code"""
    st.write("Generating code")
    result = llm.invoke(state["input"])
    return {"output": result.content}

def generate_test_case(state: State):
    """Generate test case"""
    result = llm.invoke(state["input"])
    test_passed = True  # Simulated test result
    return {"output": result.content, "test_result": test_passed}

def router(state: State):
    """Route the input to the appropriate node"""
    decision = llm_router.invoke(
        [
            SystemMessage(content="Route the input to generate code or review the code."),
            HumanMessage(content=state["input"]),
        ]
    )
    return {"decision": decision.content}

# Conditional edge functions
def route_decision(state: State):
    if "generate code" in state["decision"].lower():
        return "generate_code"
    elif "review the code" in state["decision"].lower():
        return "review"
    return "generate_code"  # Default case

def route_test_result(state: State):
    if state["test_result"]:
        return "end"
    return "generate_code"

def route_review_result(state: State):
    if state["review_result"]:
        return "generate_test_case"
    return "generate_code"

# Build workflow
router_builder = StateGraph(State)
router_builder.add_node("generate_code", generate_code)
router_builder.add_node("review", review)
router_builder.add_node("generate_test_case", generate_test_case)
router_builder.add_node("router", router)

router_builder.add_edge(START, "router")
router_builder.add_conditional_edges(
    "router",
    route_decision,
    {"generate_code": "generate_code", "review": "review"},
)
router_builder.add_conditional_edges(
    "review",
    route_review_result,
    {"generate_test_case": "generate_test_case", "generate_code": "generate_code"}
)
router_builder.add_edge("generate_code", "generate_test_case")
router_builder.add_conditional_edges(
    "generate_test_case",
    route_test_result,
    {"end": END, "generate_code": "generate_code"}
)

# Compile workflow
router_workflow = router_builder.compile()

# Streamlit App
st.title("Code Workflow Assistant")
st.write("Enter your code or request below to review or generate code.")

# Input area
user_input = st.text_area("Input", height=200, value="""check the code for errors
def binary_search(arr, target):
    low = 0
    high = len(arr) - 1
    
    while low <= high:
        mid = (low + high) // 2
        mid_val = arr[mid]
        
        if mid_val == target:
            return mid
        elif mid_val < target:
            low = mid + 1
        else:
            high = mid - 1
    
    return -1""")

# Button to process input
if st.button("Process"):
    initial_state = {"input": user_input}
    try:
        final_state = router_workflow.invoke(initial_state)
        st.subheader("Output")
        st.write(final_state["output"])
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

# Optional: Display workflow graph (requires additional setup)
if st.checkbox("Show Workflow Graph"):
    from IPython.display import Image
    st.image(router_workflow.get_graph().draw_mermaid_png(), caption="Workflow Graph") # type: ignore