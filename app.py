"""
Step 4 — Streamlit chat interface. Connects all the pieces:

    user message
        │
        ▼
    guardrails.check_guardrails()  ── rule fires ──► fixed safety reply (AI NOT called)
        │ safe
        ▼
    agent.run_agent()  ── uses rag.retrieve() + tools.py ──► answer + tool log

Run with:   streamlit run app.py
"""
import os
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from guardrails import check_guardrails

st.set_page_config(page_title="Dental Screening Assistant", page_icon="🦷")

# ---------------------------------------------------------------------------
# Header + disclaimer (required by the guidebook)
# ---------------------------------------------------------------------------
st.title("🦷 Dental Screening Assistant")
st.warning(
    "**This is a screening assistance tool, not a diagnostic tool.** "
    "It explains your screening report and gives general guidance. "
    "Always confirm with a dentist. In an emergency, call **995** or go to **A&E**."
)

# Stop early with a clear message if the key isn't set
if "GEMINI_API_KEY" not in os.environ:
    st.error("GEMINI_API_KEY is not set. In the terminal run: "
             "`set GEMINI_API_KEY=your-key` then `streamlit run app.py`.")
    st.stop()

from agent import run_agent  # imported after the key check (agent.py reads the key on import)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
FAQS = [
    "How much does a cleaning cost with CHAS Blue?",
    "My report says moderate plaque, what should I buy?",
    "I live in Jurong and have a CHAS card, where can I go?",
    "What does Advanced gum recession mean?",
    "My face is swelling",            # safety test
    "What antibiotic should I take?",  # safety test
]

faq_clicked = None  # set to the question text if a FAQ button is pressed
with st.sidebar:
    st.header("Try asking")
    st.caption("Click a question to ask it.")
    for i, q in enumerate(FAQS):
        # use_container_width makes every button the same full width
        if st.button(q, key=f"faq_{i}", use_container_width=True):
            faq_clicked = q
    st.divider()
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

# ---------------------------------------------------------------------------
# Chat history
#   messages = what we SHOW: {"role", "content", "tools", "guardrail"}
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []


def show_extras(msg):
    """Under an assistant reply, show which guardrail or tools were used."""
    if msg.get("guardrail"):
        st.caption(f"🛡️ Safety rule triggered: **{msg['guardrail']}**. The AI was not called.")
    if msg.get("tools"):
        with st.expander(f"🔧 Tools used ({len(msg['tools'])})"):
            for t in msg["tools"]:
                st.markdown(f"**{t['tool']}**  `{t['args']}`")
                st.code(t["result"], language=None)


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        show_extras(msg)


def build_history(max_turns: int = 6):
    """Convert recent chat into LangChain messages so the agent has context.
    Guardrail replies are skipped: they weren't produced by the model."""
    history = []
    for m in st.session_state.messages[-max_turns * 2:]:
        if m.get("guardrail"):
            continue
        cls = HumanMessage if m["role"] == "user" else AIMessage
        history.append(cls(m["content"]))
    return history


# ---------------------------------------------------------------------------
# Handle a new message
# ---------------------------------------------------------------------------
# A message can come from the chat box OR a clicked FAQ button
typed = st.chat_input("Ask about your screening report, costs, or products...")
if prompt := (typed or faq_clicked):
    history = build_history()  # take history BEFORE adding the new message

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # 1) Rule-first: guardrails
        hit = check_guardrails(prompt)
        if hit:
            rule, reply = hit
            msg = {"role": "assistant", "content": reply, "guardrail": rule}
        # 2) Model-second: the agent
        else:
            with st.spinner("Thinking..."):
                try:
                    answer, tool_log = run_agent(prompt, history)
                except Exception as e:
                    answer, tool_log = f"Sorry, something went wrong: `{e}`", []
            msg = {"role": "assistant", "content": answer, "tools": tool_log}

        st.markdown(msg["content"])
        show_extras(msg)
        st.session_state.messages.append(msg)