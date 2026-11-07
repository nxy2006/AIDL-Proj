"""
Agent tools for the dental screening assistant (Step 2).

All numbers come from Dental Cost Info.txt / ProductRecommendation.txt.
They are hard-coded as dicts on purpose: the LLM never does the maths
or guesses a figure, it only decides WHICH tool to call.
"""
from typing import Literal, Optional
from langchain_core.tools import tool

# ---------------------------------------------------------------------------
# DATA — MOH 2025 fee benchmarks, private primary care, EXCLUDING GST (S$)
# ---------------------------------------------------------------------------
BENCHMARKS = {
    "consultation": (21, 31),
    "scaling": (35, 60),
    "polishing": (26, 38),
    "xray": (16, 33),
    "filling_simple": (65, 160),
    "filling_complex": (80, 180),
    "extraction_anterior": (70, 120),
    "extraction_posterior": (93, 164),
    "root_canal_anterior": (400, 775),
    "root_canal_premolar": (500, 900),
    "root_canal_molar": (872, 1400),
    "crown": (750, 1400),
    "denture_complete": (500, 818),
    "denture_partial_simple": (320, 603),
    "denture_partial_complex": (450, 750),
    "denture_reline": (80, 150),
}

# CHAS / PG / MG maximum subsidy per claim (effective 1 Oct 2025), S$
_CAP_ORDER = [
    "consultation", "polishing", "scaling", "xray",
    "extraction_anterior", "extraction_posterior",
    "filling_simple", "filling_complex", "denture_reline", "crown",
    "denture_complete", "denture_partial_simple", "denture_partial_complex",
    "root_canal_anterior", "root_canal_premolar", "root_canal_molar",
]
_CAP_VALUES = {
    "pioneer":  [30.5, 30.5, 40, 21, 38.5, 78.5, 40, 60, 85, 625, 418.5, 314, 395.5, 336, 472.5, 594.5],
    "merdeka":  [25.5, 25.5, 35, 16, 33.5, 73.5, 35, 55, 80, 620, 413.5, 309, 390.5, 331, 467.5, 589.5],
    "chas_blue":   [20.5, 20.5, 30, 11, 28.5, 68.5, 30, 50, 75, 615, 408.5, 304, 385.5, 326, 462.5, 584.5],
    "chas_orange": [13.5, 13.5, 20, 7.5, 19, 45.5, 20, 33.5, 50, 410, 272.5, 202.5, 257, 217.5, 308.5, 389.5],
}
CHAS_CAPS = {card: dict(zip(_CAP_ORDER, vals)) for card, vals in _CAP_VALUES.items()}

# Procedures NOT covered by the MOH benchmark or CHAS (hospital 2024 averages + Medisave)
HOSPITAL_ONLY = {
    "implant": {
        "range": (4539, 6600),
        "medisave": "Medisave: up to S$1,950 per implant insertion + up to S$300 surgical consumables",
        "note": "Implant + crown per unit at restructured hospitals. Bone grafting / sinus lift cost extra.",
    },
    "wisdom_tooth_surgery": {
        "range": (784, 1319),
        "medisave": "Medisave: up to S$1,250 (1 tooth) or S$2,150 (2-3 teeth) + up to S$300 consumables",
        "note": "Surgical removal of one impacted wisdom tooth under local anaesthesia.",
    },
}

# "cleaning" is billed as two line items
COMBOS = {"cleaning": ["scaling", "polishing"]}

ProcedureName = Literal[
    "cleaning", "consultation", "scaling", "polishing", "xray",
    "filling_simple", "filling_complex",
    "extraction_anterior", "extraction_posterior",
    "root_canal_anterior", "root_canal_premolar", "root_canal_molar",
    "crown", "denture_complete", "denture_partial_simple",
    "denture_partial_complex", "denture_reline",
    "implant", "wisdom_tooth_surgery",
]
CardType = Literal["none", "chas_blue", "chas_orange", "merdeka", "pioneer"]


# ---------------------------------------------------------------------------
# TOOL 1 — estimate_cost
# ---------------------------------------------------------------------------
@tool
def estimate_cost(procedure: ProcedureName, card_type: CardType = "none") -> str:
    """Estimate the cost of a dental procedure in Singapore and the out-of-pocket
    amount after CHAS / Pioneer Generation / Merdeka Generation subsidy.
    Use this whenever the user asks how much a procedure costs or how much
    their subsidy card covers. If the user has not said which card they have,
    use card_type='none'. 'cleaning' means scaling + polishing."""
    if procedure in HOSPITAL_ONLY:
        info = HOSPITAL_ONLY[procedure]
        lo, hi = info["range"]
        return (
            f"{procedure}: S${lo:,}–S${hi:,} (restructured hospital 2024 average, excl. GST). "
            f"{info['note']} Not covered by CHAS. {info['medisave']}. "
            "Figures are reference estimates, not a quote."
        )

    parts = COMBOS.get(procedure, [procedure])
    lo = sum(BENCHMARKS[p][0] for p in parts)
    hi = sum(BENCHMARKS[p][1] for p in parts)
    result = f"{procedure}: MOH benchmark S${lo:,}–S${hi:,} at a private clinic (excl. GST)."

    if card_type != "none":
        cap = sum(CHAS_CAPS[card_type][p] for p in parts)
        pay_lo, pay_hi = max(lo - cap, 0), max(hi - cap, 0)
        result += (
            f" {card_type} subsidy: up to S${cap:,.2f} at a CHAS clinic. "
            f"Estimated out-of-pocket: S${pay_lo:,.2f}–S${pay_hi:,.2f} (before GST)."
        )

    result += (
        " Consultation and X-rays are usually billed separately; 9% GST may be added. "
        "Reference estimate only, not a quote."
    )
    return result


# ---------------------------------------------------------------------------
# TOOL 2 — find_clinic (MOCK DATA for the prototype)
# ---------------------------------------------------------------------------
MOCK_CLINICS = [
    {"name": "Demo Dental Clinic AMK", "area": "north", "type": "chas"},
    {"name": "Demo Smile Studio Woodlands", "area": "north", "type": "private"},
    {"name": "Demo Dental Tampines", "area": "east", "type": "chas"},
    {"name": "Demo Family Dental Bedok", "area": "east", "type": "chas"},
    {"name": "Demo Dental Jurong", "area": "west", "type": "chas"},
    {"name": "Demo Polyclinic Dental (West)", "area": "west", "type": "polyclinic"},
    {"name": "Demo Dental Novena", "area": "central", "type": "private"},
    {"name": "National Dental Centre Singapore (Outram)", "area": "central", "type": "hospital"},
]


@tool
def find_clinic(
    area: Literal["north", "south", "east", "west", "central"],
    clinic_type: Literal["any", "chas", "private", "polyclinic", "hospital"] = "any",
) -> str:
    """Find dental clinics in a region of Singapore. Use when the user asks where
    to go for treatment or for a clinic near them. Use clinic_type='chas' if the
    user has a CHAS, Pioneer or Merdeka Generation card. Use 'hospital' for
    implants, wisdom tooth surgery or other specialist work."""
    matches = [
        c for c in MOCK_CLINICS
        if c["area"] == area and (clinic_type == "any" or c["type"] == clinic_type)
    ]
    if not matches:
        return (f"No demo clinics listed for {area} / {clinic_type}. "
                "Use the official locator at https://www.chas.sg for CHAS clinics.")
    lines = [f"- {c['name']} ({c['type']})" for c in matches]
    return ("Clinics (prototype demo data):\n" + "\n".join(lines) +
            "\nFor a full, current list of CHAS clinics see https://www.chas.sg")


# ---------------------------------------------------------------------------
# TOOL 3 — recommend_product
# ---------------------------------------------------------------------------
# Product CATEGORIES (not brands), from ProductRecommendation.txt.
# Prescription items (chlorhexidine, 5000ppm toothpaste) are deliberately
# phrased as "ask your dentist" so this tool never recommends a medication.
PRODUCTS = {
    "plaque": {
        "initial": ["Soft-bristle toothbrush (manual is fine)", "Standard fluoride toothpaste (1,000–1,500 ppm)",
                    "Dental floss or interdental brushes, used daily"],
        "moderate": ["Electric toothbrush with 2-min timer and pressure sensor", "Standard fluoride toothpaste",
                     "Interdental brushes (buy a variety pack to find your sizes)"],
        "advanced": ["Electric toothbrush with pressure sensor", "Interdental brushes",
                     "Alcohol-free antiseptic mouthwash (CPC or essential-oil) as an add-on — "
                     "ask your dentist before using any prescription rinse"],
    },
    "gum_recession": {
        "initial": ["Soft or extra-soft toothbrush", "Standard fluoride toothpaste"],
        "moderate": ["Extra-soft or electric toothbrush with pressure sensor / sensitive mode",
                     "Desensitising toothpaste (use daily for 2–4 weeks)", "Interdental brushes",
                     "Alcohol-free fluoride mouthwash, used at a different time from brushing"],
        "advanced": ["Electric toothbrush with pressure sensor / sensitive mode", "Desensitising toothpaste",
                     "Interdental brushes", "Alcohol-free fluoride mouthwash"],
    },
    "caries": {
        "any": ["Standard fluoride toothpaste — spit, don't rinse",
                "Daily fluoride mouthwash (~225 ppm), used mid-day",
                "Ask your dentist whether a higher-strength fluoride toothpaste suits you"],
    },
    "gum_inflammation": {
        "any": ["Soft-bristle or electric toothbrush", "Dental floss or interdental brushes, used daily",
                "Alcohol-free antiseptic mouthwash as an add-on (not a replacement for brushing)"],
    },
    "missing_tooth": {
        "any": ["No product replaces a missing tooth — book a tooth replacement consult",
                "If you already have a bridge or implant: interdental brushes or a water flosser"],
    },
}


@tool
def recommend_product(
    finding: Literal["plaque", "gum_recession", "caries", "gum_inflammation", "missing_tooth"],
    severity: Optional[Literal["initial", "moderate", "advanced"]] = None,
) -> str:
    """Recommend oral-care product categories for a finding on the user's
    screening report. Use when the user asks what toothbrush, toothpaste,
    floss or mouthwash to buy. 'caries' = cavity. If severity is unknown,
    leave it empty."""
    options = PRODUCTS[finding]
    if "any" in options:
        items = options["any"]
    else:
        items = options[severity or "initial"]
    return (f"Suggested product types for {finding}" + (f" ({severity})" if severity else "") + ":\n"
            + "\n".join(f"- {i}" for i in items)
            + "\nThese support home care; they don't replace a dental visit.")


TOOLS = [estimate_cost, find_clinic, recommend_product]