"""
Step 3 — Guardrails: a rule-based safety layer that runs BEFORE the LLM.

"Rule-first, model-second": if a message matches a safety rule, we return a
fixed, pre-written reply and the LLM is never called. This is deterministic,
so it behaves the same every time — unlike relying on the prompt alone.

Usage (in app.py):
    reply = check_guardrails(user_text)
    if reply:            # a rule fired -> show this, skip the agent
        ...
    else:                # safe -> call run_agent()
        ...
"""
import re

# ---------------------------------------------------------------------------
# Fixed replies
# ---------------------------------------------------------------------------
EMERGENCY_REPLY = (
    "🚨 This may be a dental or medical emergency. "
    "Please call **995** or go to the nearest **A&E** immediately. "
    "Do not wait for a routine dental appointment."
)

MEDICATION_REPLY = (
    "I can't recommend medications, antibiotics or dosages. "
    "Please consult a dentist, doctor or pharmacist about medication."
)

DIAGNOSIS_REPLY = (
    "I can't diagnose conditions. This tool only helps you understand your "
    "screening report. Please see a dentist for a proper examination."
)

# ---------------------------------------------------------------------------
# Rules — checked in this order. Emergency comes first because it matters most.
# Each pattern is a regex; \b = word boundary so "swell" won't match "swellness".
# ---------------------------------------------------------------------------
EMERGENCY_PATTERNS = [
    r"\bswell(ing|ed|s)?\b", r"\bswollen\b",
    r"bleeding (won'?t|will not|doesn'?t|does not|can'?t) stop",
    r"\b(non[- ]?stop|uncontrolled|heavy|severe) bleeding\b",
    r"\bcan'?t stop (the )?bleeding\b",
    r"\b(can'?t|cannot|hard to|difficulty|trouble) (breath|breathe|breathing|swallow|swallowing)\b",
    r"\btrauma\b", r"\bknocked[- ]out\b", r"\b(broken|fractured) jaw\b",
    r"\b(car |road )?accident\b", r"\bhit (in|on) the (face|mouth|jaw)\b",
    r"\b(abscess|pus)\b.*\bfever\b", r"\bfever\b.*\b(abscess|pus|swelling)\b",
    r"\bunbearable pain\b", r"\bsevere pain\b",
]

MEDICATION_PATTERNS = [
    r"\bantibiotics?\b", r"\bmedications?\b", r"\bmedicines?\b",
    r"\bpainkillers?\b", r"\bpain ?relief\b", r"\bpills?\b", r"\bdos(e|age|ing)\b",
    r"\bprescri(be|ption)\b", r"\bwhat (drug|tablet)s?\b",
    r"\b(amoxicillin|ibuprofen|paracetamol|panadol|nurofen|metronidazole|"
    r"augmentin|clindamycin|aspirin|codeine|chlorhexidine)\b",
]

DIAGNOSIS_PATTERNS = [
    r"\bdiagnos(e|is)\b", r"\bdo i have\b", r"\bam i (sick|infected)\b",
    r"\bis (it|this) (cancer|infected|an infection)\b", r"\bwhat disease\b",
]

RULES = [
    ("emergency", EMERGENCY_PATTERNS, EMERGENCY_REPLY),
    ("medication", MEDICATION_PATTERNS, MEDICATION_REPLY),
    ("diagnosis", DIAGNOSIS_PATTERNS, DIAGNOSIS_REPLY),
]


def check_guardrails(text: str):
    """Return (rule_name, reply) if a safety rule fires, otherwise None."""
    t = text.lower()
    for name, patterns, reply in RULES:
        if any(re.search(p, t) for p in patterns):
            return name, reply
    return None


# ---------------------------------------------------------------------------
# Self-test: python guardrails.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    tests = [
        # (message, expected rule or None)
        ("My face is swelling after my tooth hurt", "emergency"),
        ("The bleeding won't stop after extraction", "emergency"),
        ("I can't breathe properly and my jaw hurts", "emergency"),
        ("My tooth got knocked out playing football", "emergency"),
        ("What antibiotic should I take for my tooth?", "medication"),
        ("How much panadol can I take?", "medication"),
        ("Do I have gum disease?", "diagnosis"),
        ("How much does scaling cost?", None),
        ("What toothbrush should I buy for gum recession?", None),
        ("What does Moderate plaque mean on my report?", None),
    ]
    passed = 0
    for msg, expected in tests:
        result = check_guardrails(msg)
        got = result[0] if result else None
        ok = got == expected
        passed += ok
        print(f"{'PASS' if ok else 'FAIL'}  [{got or 'safe':10}] {msg}")
    print(f"\n{passed}/{len(tests)} passed")