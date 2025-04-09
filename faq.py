from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
from typing_extensions import TypedDict

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY environment variable not set.")

# Initialize LLM
llm = ChatGroq(model="qwen-2.5-32b", api_key=groq_api_key)
llm_router = ChatGroq(model="qwen-2.5-32b", api_key=groq_api_key)

# Define State
class State(TypedDict):
    input: str
    decision: str
    output: str
    test_result: bool
    review_result: bool

# Nodes
def review(state: State) -> Dict:
    """Review the code"""
    result = llm.invoke([HumanMessage(content=state["input"])])
    review_passed = True  # Simulated review result
    return {"output": result.content, "review_result": review_passed}

def generate_code(state: State) -> Dict:
    """Generate code"""
    result = llm.invoke([HumanMessage(content=state["input"])])
    return {"output": result.content}

def generate_test_case(state: State) -> Dict:
    """Generate test case"""
    result = llm.invoke([HumanMessage(content=state["input"])])
    test_passed = True  # Simulated test result
    return {"output": result.content, "test_result": test_passed}

def router(state: State) -> Dict:
    """Route the input to the appropriate node"""
    decision = llm_router.invoke(
        [
            SystemMessage(content="Route the input to generate code, review the code, or provide a generic answer."),
            HumanMessage(content=state["input"]),
        ]
    )
    return {"decision": decision.content}

def generic_answer(state: State) -> Dict:
    """Provide a generic answer"""
    return {"output": "This is a generic response to your input. How can I assist you further?"}

def faq(state: State) -> Dict:
    """Provide FAQ information"""
    faq_text = """
    **Frequently Asked Questions:**
    1. **What can this app do?** It can review code, generate code, or provide generic answers.
    2. **How do I use it?** Send a POST request to /process with a JSON body containing your input.
    3. **What if I get an error?** Check your input or contact support.
    """
    return {"output": state.get("output", "") + "\n\n" + faq_text}

# Conditional edge functions
def route_decision(state: State) -> str:
    decision_lower = state["decision"].lower()
    if "generate code" in decision_lower:
        return "generate_code"
    elif "review the code" in decision_lower:
        return "review"
    elif "generic answer" in decision_lower:
        return "generic_answer"
    return "generate_code"  # Default case

def route_test_result(state: State) -> str:
    if state["test_result"]:
        return "end"
    return "generate_code"

def route_review_result(state: State) -> str:
    if state["review_result"]:
        return "generate_test_case"
    return "generate_code"

# Build workflow
router_builder = StateGraph(State)
router_builder.add_node("generate_code", generate_code)
router_builder.add_node("review", review)
router_builder.add_node("generate_test_case", generate_test_case)
router_builder.add_node("router", router)
router_builder.add_node("generic_answer", generic_answer)
router_builder.add_node("faq", faq)

# Add edges
router_builder.add_edge(START, "router")
router_builder.add_conditional_edges(
    "router",
    route_decision,
    {
        "generate_code": "generate_code",
        "review": "review",
        "generic_answer": "generic_answer"
    },
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
router_builder.add_edge("generic_answer", "faq")
router_builder.add_edge("faq", END)

# Compile workflow
router_workflow = router_builder.compile()

# FastAPI app
app = FastAPI(
    title="Code Workflow API",
    description="API for code generation, review, testing, and FAQs using LangGraph and Groq LLM",
    version="1.0.0",
)

# Pydantic model for request
class CodeRequest(BaseModel):
    input: str

# Pydantic model for response
class WorkflowResponse(BaseModel):
    output: str
    decision: Optional[str] = None
    test_result: Optional[bool] = None
    review_result: Optional[bool] = None
    error: Optional[str] = None

@app.post("/process", response_model=WorkflowResponse)
async def process_code(request: CodeRequest):
    """Process the input to generate, review, test code, or provide a generic answer."""
    initial_state = {"input": request.input}
    try:
        final_state = router_workflow.invoke(initial_state)
        return WorkflowResponse(
            output=final_state["output"],
            decision=final_state.get("decision"),
            test_result=final_state.get("test_result"),
            review_result=final_state.get("review_result")
        )
    except Exception as e:
        return WorkflowResponse(output="", error=str(e))

@app.get("/health")
async def health_check():
    """Check if the API is running."""
    return {"status": "healthy"}

@app.get("/docs")
async def get_docs():
    """Provide API documentation link."""
    return {"message": "Visit /docs in your browser for interactive API documentation."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)