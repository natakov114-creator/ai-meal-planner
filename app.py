import os
import re
from io import StringIO, BytesIO
from typing import List, Optional

import pandas as pd
import streamlit as st
from openai import OpenAI

# =========================
# OpenAI client via env/secrets (DO NOT hardcode your key)
# =========================
api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", "")
if not api_key or not api_key.startswith("sk-"):
    st.error("OPENAI_API_KEY not found. Add it to environment variables or Streamlit secrets.")
    st.stop()
client = OpenAI(api_key=api_key)

# =========================
# Nutrition guardrails (expand as needed)
# =========================
DISCRETIONARY_KEYWORDS = [
    "fried", "deep-fried", "battered", "donut", "pastry", "croissant",
    "chips", "fries", "ice cream", "candy", "chocolate bar", "soda",
    "alcohol", "beer", "wine", "cocktail", "sausage roll"
]

RULES = {
    "diabetes": {
        "avoid": [
            "white bread", "white rice", "sugary cereal", "pastry", "donut",
            "fruit juice", "regular soda", "added sugar", "syrup"
        ],
        "prefer": [
            "whole grain", "brown rice", "oats", "wholemeal", "legumes", "lentils",
            "berries", "leafy greens", "non-starchy vegetables", "nuts", "seeds"
        ],
        "swaps": {
            "white rice": "brown rice",
            "white bread": "whole-grain bread",
            "sugary cereal": "oats or low-sugar muesli",
            "fruit juice": "whole fruit",
            "regular soda": "sparkling water or diet soda",
            "syrup": "berries",
            "pastry": "wholegrain toast with avocado"
        }
    },
    "hypertension": {},
    "cholesterol": {},
    "ckd": {},
    "ibs": {},
    "gerd": {},
}

INTOLERANCE_RULES = {
    "lactose": {
        "avoid": ["milk", "ice cream", "soft cheese", "cream", "yoghurt", "yogurt"],
        "swaps": {
            "milk": "lactose-free milk or almond milk",
            "yogurt": "lactose-free yogurt or coconut yogurt",
            "soft cheese": "hard cheese or lactose-free cheese",
            "ice cream": "sorbet or lactose-free ice cream",
        },
    },
    "gluten": {
        "avoid": ["wheat", "barley", "rye", "bulgur", "couscous", "semolina", "seitan", "farro"],
        "swaps": {
            "wheat bread": "gluten-free bread",
            "pasta": "gluten-free pasta",
            "couscous": "quinoa",
            "bulgur": "quinoa",
            "farro": "brown rice",
        },
    },
    "sorbitol": {
        "avoid": ["pear", "stone fruit skins", "sorbitol-sweetened gum", "diet candy"],
        "swaps": {"pear": "apple or berries"},
    },
}

def _contains_any(text: str, words: List[str]) -> bool:
    t = text.lower()
    return any(w in t for w in words)

def _first_match(text: str, words: List[str]) -> Optional[str]:
    t = text.lower()
    for w in words:
        if w in t:
            return w
    return None

# =========================
# GPT: Generate a 1-day plan (Markdown table)
# =========================
def generate_meal_plan(
    calories: int, protein: int, diet="omnivore", window="13:00-21:00",
    preferences=None, exclusions=None, intolerances=None, condition=None
) -> str:
    preferences  = preferences  or []
    exclusions   = exclusions   or []
    intolerances = intolerances or []

    prompt = f"""
    Create a one-day meal plan:
    - Calories: ~{calories}
    - Protein: ≥{protein} g
    - Diet: {diet}
    - Eating window: {window}

    Adjustments:
    - Food preferences (prioritize): {", ".join(preferences) if preferences else "None"}
    - Exclusions (must not include): {", ".join(exclusions) if exclusions else "None"}
    - Intolerances/allergies: {", ".join(intolerances) if intolerances else "None"}
    - Medical condition: {condition if condition else "None"}

    For each item:
    - Respect preferences, exclusions, and intolerances.
    - If condition is diabetes: prefer low-GI whole grains, legumes, non-starchy vegetables; avoid added sugars, juice, white breads/rice.
    - If condition is hypertension: prefer low-sodium choices; avoid processed meats and salty sauces.
    - If condition is cholesterol: reduce saturated fat; avoid fried foods; prefer fish, legumes, nuts, olive oil.
    - If item is discretionary (sweets, alcohol, deep-fried, sugary drinks), mark with ⚠️ in a "Note" column.

    Return ONLY a Markdown table with columns:
    | Meal | Food | Portion | Calories | Protein | Note |
    """
    r = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=900,
        temperature=0.5
    )
    return r.choices[0].message.content

# =========================
# Parse Markdown → DataFrame
# =========================
def markdown_to_df(md_text: str) -> pd.DataFrame:
    md_text = md_text.strip().replace("```", "")
    lines = [ln for ln in md_text.splitlines() if "|" in ln and "---" not in ln]
    if not lines:
        raise ValueError("No Markdown table detected.")
    table_text = "\n".join(lines)
    df = pd.read_csv(StringIO(table_text), sep="|", engine="python").dropna(axis=1, how="all")
    df = df.apply(lambda s: s.str.strip() if s.dtype == "object" else s)
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    df.columns = [c.strip() for c in df.columns]
    for col in ("Calories", "Protein"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for must in ("Meal", "Food", "Portion", "Calories", "Protein", "Note"):
        if must not in df.columns:
            df[must] = "" if must in ("Meal", "Food", "Portion", "Note") else pd.NA
    return df

# =========================
# Guardrails: swaps/flags
# =========================
def apply_guardrails(df: pd.DataFrame, condition=None, intolerances=None, exclusions=None, preferences=None) -> pd.DataFrame:
    condition    = (condition or "").lower() or None
    intolerances = [t.lower() for t in (intolerances or [])]
    exclusions   = [e.lower() for e in (exclusions or [])]
    preferences  = [p.lower() for p in (preferences or [])]

    def try_swap(food_text, swaps_dict):
        t = food_text
        for bad, good in swaps_dict.items():
            if bad in t.lower():
                t = re.sub(bad, good, t, flags=re.IGNORECASE)
        return t

    notes = []
    fixed_foods = []

    for _, row in df.iterrows():
        food = str(row.get("Food", "") or "")
        note_msgs = []

        # discretionary flag
        if _contains_any(food, DISCRETIONARY_KEYWORDS):
            note_msgs.append("⚠️ Discretionary food – limit intake")

        # exclusions
        ex_hit = _first_match(food, exclusions)
        if ex_hit:
            replacement = (preferences[0] + " (swap)") if preferences else "alternative lean protein/wholegrain (swap)"
            food = re.sub(ex_hit, replacement, food, flags=re.IGNORECASE)
            note_msgs.append(f"Excluded item '{ex_hit}' swapped")

        # intolerances
        for itol in intolerances:
            rules = INTOLERANCE_RULES.get(itol)
            if not rules:
                continue
            bad = _first_match(food, rules.get("avoid", []))
            if bad:
                food = try_swap(food, rules.get("swaps", {}))
                note_msgs.append(f"{itol.capitalize()} swap for '{bad}'")

        # condition rules (simple heuristics)
        if condition == "diabetes":
            dr = RULES["diabetes"]
            bad = _first_match(food, dr["avoid"])
            if bad:
                food = try_swap(food, dr["swaps"])
                note_msgs.append(f"Diabetes-friendly swap for '{bad}'")
        elif condition == "hypertension":
            if _contains_any(food, ["soy sauce", "processed meat", "bacon", "salami", "ramen seasoning"]):
                food = re.sub("soy sauce", "reduced-sodium soy sauce", food, flags=re.IGNORECASE)
                note_msgs.append("Lower-sodium swap")
        elif condition == "cholesterol":
            if _contains_any(food, ["fried", "butter", "cream", "fatty cuts"]):
                food = food.replace("fried", "grilled")
                food = re.sub("butter|cream", "olive oil or yogurt", food, flags=re.IGNORECASE)
                note_msgs.append("Lower-saturated-fat swap")

        fixed_foods.append(food)
        notes.append("; ".join(dict.fromkeys(note_msgs)))

    df["Food"] = fixed_foods
    df["Note"] = df["Note"].fillna("").astype(str)
    df["Note"] = (df["Note"].str.strip() + "; " + pd.Series(notes)).str.strip("; ").str.strip()
    return df

# =========================
# Styled PDF export → bytes (for download)
# =========================
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet

def build_pdf_bytes(df: pd.DataFrame, client_name="Client", calories=0, protein=0, diet="omnivore", window="", condition=None, intolerances=None) -> bytes:
    buf = BytesIO()
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(buf, pagesize=A4)
    story = []

    story.append(Paragraph("<b>1-Day Meal Plan</b>", styles["Title"]))
    story.append(Paragraph(f"{client_name} • {diet} • {window}", styles["Normal"]))
    story.append(Spacer(1, 12))

    kcals = pd.to_numeric(df["Calories"], errors="coerce").sum()
    prots = pd.to_numeric(df["Protein"], errors="coerce").sum()
    story.append(Paragraph(f"Target: ~{calories} kcal • ≥{protein} g protein", styles["Normal"]))
    story.append(Paragraph(f"Planned: {int(kcals)} kcal • {int(prots)} g protein", styles["Normal"]))
    story.append(Spacer(1, 12))

    table_data = [list(df.columns)] + df.astype(str).values.tolist()
    t = Table(table_data, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#DDEEFF")),
        ("GRID", (0,0), (-1,-1), 0.25, colors.black),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("ALIGN", (0,0), (-1,-1), "CENTER"),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#F5F9FF")]),
    ]))
    story.append(t)
    story.append(Spacer(1, 18))

    if condition or (intolerances and len(intolerances) > 0):
        story.append(Paragraph("<b>Special Considerations</b>", styles["Heading2"]))
        if condition:
            story.append(Paragraph(f"Medical condition: <b>{condition}</b>", styles["Normal"]))
        if intolerances:
            story.append(Paragraph(f"Intolerances/allergies considered: <b>{', '.join(intolerances)}</b>", styles["Normal"]))
        story.append(Spacer(1, 8))

    doc.build(story)
    pdf_bytes = buf.getvalue()
    buf.close()
    return pdf_bytes

# =========================
# UI
# =========================
st.set_page_config(page_title="AI Meal Planner", page_icon="🥗", layout="centered")
st.title("🥗 AI Meal Planner")

with st.form("plan_form"):
    col1, col2 = st.columns(2)
    with col1:
        calories = st.number_input("Calories (target)", min_value=1000, max_value=4000, value=1600, step=50)
        diet = st.selectbox("Diet", ["omnivore", "vegetarian", "vegan", "pescatarian"])
        client_name = st.text_input("Client name", "Client")
    with col2:
        protein = st.number_input("Protein (g, minimum)", min_value=50, max_value=250, value=110, step=5)
        window = st.text_input("Eating window", "13:00-21:00")
        condition = st.selectbox("Medical condition (optional)", ["none", "diabetes", "hypertension", "cholesterol", "ckd", "ibs", "gerd"])

    preferences = st.text_input("Food preferences (comma-separated)", "salmon, oats, spinach, greek yogurt")
    exclusions  = st.text_input("Exclusions / dislikes (comma-separated)", "pork, pear")
    intolerances = st.multiselect("Intolerances/allergies", ["lactose", "gluten", "sorbitol"])

    submit = st.form_submit_button("Generate Plan")

if submit:
    prefs = [p.strip() for p in preferences.split(",") if p.strip()]
    excl  = [e.strip() for e in exclusions.split(",") if e.strip()]
    cond  = None if condition == "none" else condition

    with st.spinner("Generating meal plan..."):
        md = generate_meal_plan(
            calories, protein, diet=diet, window=window,
            preferences=prefs, exclusions=excl,
            intolerances=intolerances, condition=cond
        )
        try:
            df = markdown_to_df(md)
        except Exception as e:
            st.error(f"Could not parse the model output into a table. Error: {e}")
            st.code(md)
            st.stop()

        df = apply_guardrails(df, condition=cond, intolerances=intolerances, exclusions=excl, preferences=prefs)

        st.subheader("Plan Preview")
        st.dataframe(df, use_container_width=True)

        total_kcal = int(pd.to_numeric(df["Calories"], errors="coerce").sum())
        total_prot = int(pd.to_numeric(df["Protein"], errors="coerce").sum())
        st.write(f"**Totals:** {total_kcal} kcal • {total_prot} g protein")

        # Downloads
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download CSV", data=csv_bytes, file_name="Meal_Plan.csv", mime="text/csv")

        pdf_bytes = build_pdf_bytes(
            df, client_name=client_name, calories=calories, protein=protein,
            diet=diet, window=window, condition=cond, intolerances=intolerances
        )
        st.download_button("⬇️ Download PDF", data=pdf_bytes, file_name="Meal_Plan.pdf", mime="application/pdf")

st.caption("Tip: add your OPENAI_API_KEY in Streamlit secrets after deploying. Review plans clinically when medical conditions are selected.")
