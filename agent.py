"""
Agent loop (Step 2): LLM decides which tool to call, we run it, feed the
result back, and repeat until the LLM gives a final answer.

Integration points with your friend's work:
  - rag.py        -> retrieve(query: str) -> str       (Step 1)
  - guardrails.py -> check_guardrails(text) -> str|None (Step 3, used in app.py)
"""
import warnings
warnings.filterwarnings("ignore", message=".*sampling parameter.*")
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from tools import TOOLS

# --- Friend's Step 1. Stub until it's ready, so you're not blocked. ---
try:
    from rag import retrieve
except ImportError:
    def retrieve(query: str) -> str:
        return "(knowledge base not connected yet)"

# Gemini is OpenAI-compatible, so ChatOpenAI works with a different base_url
llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    google_api_key=os.environ["GEMINI_API_KEY"],
    temperature=0,
)
llm_with_tools = llm.bind_tools(TOOLS)
TOOL_MAP = {t.name: t for t in TOOLS}

SYSTEM_PROMPT = """You are a dental screening assistant for Singapore users.
- For any cost, subsidy or price question, ALWAYS call estimate_cost. Never state a price from memory.
- For product questions, call recommend_product. For "where should I go", call find_clinic.
- For everything else, answer only from the CONTEXT below. If it isn't there, say you don't know.
- Never diagnose, never recommend medications, and always suggest confirming with a dentist.
- Keep answers short and in plain language.

CONTEXT:
{context}"""

MAX_STEPS = 5  # stops infinite tool loops

def to_text(content) -> str:
    """Gemini returns a list of content blocks; join just the text parts."""
    if isinstance(content, str):
        return content
    return "".join(b.get("text", "") for b in content if isinstance(b, dict))

def run_agent(user_input: str, history: list | None = None):
    """Returns (answer_text, tool_log). tool_log lets the UI show which tools ran."""
    history = history or []
    context = retrieve(user_input)
    messages = [
        SystemMessage(SYSTEM_PROMPT.format(context=context)),
        *history,
        HumanMessage(user_input),
    ]
    tool_log = []

    for _ in range(MAX_STEPS):
        ai_msg = llm_with_tools.invoke(messages)
        messages.append(ai_msg)

        if not ai_msg.tool_calls:          # no tool needed -> final answer
            return to_text(ai_msg.content), tool_log

        for call in ai_msg.tool_calls:     # run every tool the LLM asked for
            result = TOOL_MAP[call["name"]].invoke(call["args"])
            tool_log.append({"tool": call["name"], "args": call["args"], "result": result})
            messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))

    return "Sorry, I couldn't complete that. Please try rephrasing.", tool_log


if __name__ == "__main__":
    tests = [
        "How much does a cleaning cost with CHAS Blue?",
        "My report says moderate gum recession, what toothbrush should I buy?",
        "I live in Tampines and have a CHAS card, where can I go?",
        "What is plaque?",  # should NOT call a tool
    ]
    for q in tests:
        answer, log = run_agent(q)
        print("Q:", q)
        print("Tools:", [(t["tool"], t["args"]) for t in log])
        print("A:", answer, "\n" + "-" * 60)