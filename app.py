import os
import re
from io import StringIO, BytesIO
from typing import List, Optional

import pandas as pd
import streamlit as st
from openai import OpenAI

# ================================
# OpenAI client via env/secrets (DO NOT hardcode your key)
# ================================

# Prefer Streamlit Secrets, fallback to environment variables
api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")

if not api_key:
    st.error("❌ No OpenAI API key found. Please add it in Streamlit Secrets.")
    st.stop()

# Initialize OpenAI client
client = OpenAI(api_key=api_key)

# Allow choosing the model from environment or Streamlit secrets
MODEL = st.secrets.get("OPENAI_MODEL") or os.getenv("OPENAI_MODEL") or "gpt-4o-mini"

# ================================
# Nutrition guardrails (expand as needed)
# ================================
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
    }
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

# ================================
# GPT: Generate a 1-day plan (Markdown table)
# ================================
def generate_meal_plan(
    calories: int, protein: int, diet="omnivore", window="13:00-21:00",
    preferences=None, exclusions=None, intolerances=None, condition=None
) -> str:
    preferences = preferences or []
    exclusions = exclusions or []
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
- If condition is diabetes: prefer low-GI whole grains, legumes, non-starchy vegetables; avoid added sugars, juice, white bread.
- If condition is hypertension: prefer low-sodium choices; avoid processed meats and salty sauces.
- If condition is cholesterol: reduce saturated fat; avoid fried foods; prefer fish, legumes, nuts, olive oil.
- If item is discretionary (sweets, alcohol, deep-fried, sugary drinks), mark with ⚠️ in a "Note" column.

Return ONLY a Markdown table with columns:
| Meal | Food | Portion | Calories | Protein | Note |
"""

    try:
        r = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=700,
            temperature=0.5
        )
        return r.choices[0].message["content"]
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        st.error(
            f"⚠️ Error generating meal plan.\n\n"
            "Common causes:\n"
            ". Model not available for your account (try a different model)\n"
            ". Insufficient quota/billing (HTTP 429)\n"
            ". Invalid/expired API key (HTTP 401)\n\n"
            f"Details: {e}"
        )
        st.caption(tb)
        raise

# ================================
# Streamlit App UI
# ================================
st.title("🥗 AI Meal Planner")

calories = st.number_input("Daily Calories", min_value=1000, max_value=4000, value=2000, step=50)
protein = st.number_input("Daily Protein (g)", min_value=50, max_value=300, value=100, step=10)
diet = st.selectbox("Diet type", ["omnivore", "vegetarian", "vegan", "pescatarian"])
window = st.text_input("Eating window (e.g., 13:00-21:00)", value="13:00-21:00")

preferences = st.text_input("Food preferences (comma separated)").split(",") if st.text_input else []
exclusions = st.text_input("Foods to exclude (comma separated)").split(",") if st.text_input else []
intolerances = st.multiselect("Intolerances", ["lactose", "gluten", "sorbitol"])
condition = st.selectbox("Medical condition", ["None", "diabetes", "hypertension", "cholesterol"])

if st.button("Generate Meal Plan"):
    plan = generate_meal_plan(calories, protein, diet, window, preferences, exclusions, intolerances, condition)
    st.markdown(plan)

    # Export option
    pdf_bytes = plan.encode("utf-8")
    st.download_button("⬇️ Download PDF", data=pdf_bytes, file_name="Meal_Plan.pdf", mime="application/pdf")
