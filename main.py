"""
main.py  ─  Telco Churn Prediction Chatbot  (FastAPI Backend)
=============================================================
Architecture:
  • FastAPI serves the chat UI as a static HTML page (index.html)
  • POST /api/chat  ─ handles conversation with gemini-2.5-flash
  • POST /api/predict ─ runs the Random Forest model + returns explanation
  • BeautifulSoup is used server-side to parse/sanitise LLM HTML responses
    before they are forwarded to the browser

Run locally:
    uvicorn main:app --reload --port 8000
    Then open: http://localhost:8000

"""

import os, json, re, pickle, uuid
from pathlib import Path

import pandas as pd
import numpy as np
from bs4 import BeautifulSoup                 # HTML sanitisation / parsing
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from google import genai
from google.genai import types



# ══════════════════════════════════════════════════════════════════════════════
#  0.  Config
# ══════════════════════════════════════════════════════════════════════════════
BASE_DIR = Path(__file__).parent

with open(BASE_DIR / "config.json") as fh:
    config = json.load(fh)

client = genai.Client(api_key=config["GEMINI_API_KEY"])
GEMINI_MODEL = config.get("GEMINI_MODEL", "gemini-2.5-flash")
app     = FastAPI(title="Churn Prediction Chatbot API", version="1.0.0")

templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

# ══════════════════════════════════════════════════════════════════════════════
#  1.  Load ML artefacts  (loaded once at startup)
# ══════════════════════════════════════════════════════════════════════════════
with open(BASE_DIR / "customer_churn_model.pkl", "rb") as f:
    _model_data  = pickle.load(f)
ML_MODEL      = _model_data["model"]
FEATURE_NAMES = _model_data["features_names"]

with open(BASE_DIR / "encoders.pkl", "rb") as f:
    ENCODERS = pickle.load(f)

print("✅  ML model & encoders loaded.")

# ══════════════════════════════════════════════════════════════════════════════
#  2.  Feature catalogue
# ══════════════════════════════════════════════════════════════════════════════
FEATURE_META = {
    "gender":             {"type": "cat",   "values": ["Male", "Female"]},
    "Senior_Citizen ":    {"type": "int",   "values": [0, 1],  "label": "Senior Citizen"},
    "Is_Married":         {"type": "cat",   "values": ["Yes", "No"]},
    "Dependents":         {"type": "cat",   "values": ["Yes", "No"]},
    "tenure":             {"type": "int", "values": "months with company"},
    "Phone_Service":      {"type": "cat",   "values": ["Yes", "No"]},
    "Dual":               {"type": "cat",   "values": ["Yes", "No", "No phone service"]},
    "Internet_Service":   {"type": "cat",   "values": ["DSL", "Fiber optic", "No"]},
    "Online_Security":    {"type": "cat",   "values": ["Yes", "No", "No internet service"]},
    "Online_Backup":      {"type": "cat",   "values": ["Yes", "No", "No internet service"]},
    "Device_Protection":  {"type": "cat",   "values": ["Yes", "No", "No internet service"]},
    "Tech_Support":       {"type": "cat",   "values": ["Yes", "No", "No internet service"]},
    "Streaming_TV":       {"type": "cat",   "values": ["Yes", "No", "No internet service"]},
    "Streaming_Movies":   {"type": "cat",   "values": ["Yes", "No", "No internet service"]},
    "Contract":           {"type": "cat",   "values": ["Month-to-month", "One year", "Two year"]},
    "Paperless_Billing":  {"type": "cat",   "values": ["Yes", "No"]},
    "Payment_Method":     {"type": "cat",
                           "values": ["Electronic check", "Mailed check",
                                      "Bank transfer (automatic)", "Credit card (automatic)"]},
    "Monthly_Charges":    {"type": "float", "values": "monthly USD charge"},
    "Total_Charges":      {"type": "float", "values": "total USD charges"},
}

# ══════════════════════════════════════════════════════════════════════════════
#  3.  In-memory session store  {session_id: [messages]}
# ══════════════════════════════════════════════════════════════════════════════
SESSIONS: dict[str, list[dict]] = {}

# ══════════════════════════════════════════════════════════════════════════════
#  4.  System prompt
# ══════════════════════════════════════════════════════════════════════════════
SYSTEM_PROMPT = """
You are ChurnBot, a friendly and professional AI assistant for a telecom marketing team.
Your sole task: predict customer churn by collecting customer info through natural conversation.

STRICT RULES:
1. Greet the user warmly on first message and explain you need ~19 customer details.
2. Ask for ONE missing field at a time.  Be conversational, not robotic.
3. Once ALL fields are collected, output EXACTLY this structure and nothing else:

<JSON>
{
  "gender": "...",
  "Senior_Citizen ": 0,
  "Is_Married": "...",
  "Dependents": "...",
  "tenure": 0,
  "Phone_Service": "...",
  "Dual": "...",
  "Internet_Service": "...",
  "Online_Security": "...",
  "Online_Backup": "...",
  "Device_Protection": "...",
  "Tech_Support": "...",
  "Streaming_TV": "...",
  "Streaming_Movies": "...",
  "Contract": "...",
  "Paperless_Billing": "...",
  "Payment_Method": "...",
  "Monthly_Charges": 0.0,
  "Total_Charges": 0.0
}
</JSON>

NOTE: "Senior_Citizen " has a TRAILING SPACE — keep it exactly.
4. After the JSON block, output nothing else.
5. If user asks off-topic questions, answer very briefly then return to data collection.
6. Valid options per field are listed here for reference:
   gender: Male | Female
   Is_Married / Dependents / Phone_Service / Paperless_Billing: Yes | No
   Dual: Yes | No | No phone service
   Internet_Service: DSL | Fiber optic | No
   Online_Security / Online_Backup / Device_Protection / Tech_Support /
   Streaming_TV / Streaming_Movies: Yes | No | No internet service
   Contract: Month-to-month | One year | Two year
   Payment_Method: Electronic check | Mailed check |
                   Bank transfer (automatic) | Credit card (automatic)
"""

# ══════════════════════════════════════════════════════════════════════════════
#  5.  Helpers
# ══════════════════════════════════════════════════════════════════════════════

def sanitise_html(raw: str) -> str:
    """
    Use BeautifulSoup to parse LLM text that may contain stray HTML tags,
    strip dangerous attributes, and return safe inner HTML for the browser.
    Plain text is converted to paragraph-wrapped HTML.
    """
    soup = BeautifulSoup(raw, "html.parser")

    # Remove any <script> or <style> injected by the LLM
    for tag in soup.find_all(["script", "style", "iframe", "object"]):
        tag.decompose()

    # Strip event-handler attributes (onclick, onerror, etc.)
    for tag in soup.find_all(True):
        for attr in list(tag.attrs):
            if attr.lower().startswith("on"):
                del tag[attr]

    # Convert plain newlines to <br> if no block-level tags exist
    cleaned = str(soup)
    if not any(t in cleaned for t in ["<p>", "<ul>", "<ol>", "<li>", "<br"]):
        cleaned = cleaned.replace("\n", "<br>")

    return cleaned


def extract_json_block(text: str) -> dict | None:
    """Extract <JSON>…</JSON> payload from Gemini reply."""
    match = re.search(r"<JSON>(.*?)</JSON>", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass
    return None


def run_ml_prediction(raw: dict) -> dict:
    """Encode features and run the Random Forest model."""
    df = pd.DataFrame([raw])

    for col, enc in ENCODERS.items():
        if col in df.columns:
            val = df[col].iloc[0]
            known = list(enc.classes_)
            if str(val) not in known:
                val = known[0]            # safe fallback
            df[col] = enc.transform([val])

    for col in FEATURE_NAMES:
        if col not in df.columns:
            df[col] = 0
    df = df[FEATURE_NAMES]

    pred  = ML_MODEL.predict(df)[0]
    proba = ML_MODEL.predict_proba(df)[0]

    return {
        "churn":   bool(pred),
        "prob":    float(proba[1]),
        "no_prob": float(proba[0]),
        "raw":     raw,
    }


def build_explanation_prompt(result: dict) -> str:
    label    = "WILL CHURN" if result["churn"] else "WILL NOT CHURN"
    prob_pct = f"{result['prob']*100:.1f}%"
    return f"""
A telecom customer was assessed by a Random Forest churn model.

Prediction : {label}
Churn probability : {prob_pct}

Customer profile:
{json.dumps(result['raw'], indent=2)}

Write a concise explanation (4-6 bullet points using • symbol) for a NON-TECHNICAL
marketing analyst:
• Key drivers behind this prediction based on the profile.
• Specific risk or retention factors.
• One short, actionable recommendation.

Keep language warm, clear, and professional. Use plain text only — no markdown or HTML.
"""
def generate_gemini_text(messages: list[dict], temperature: float = 0.4) -> str:
    prompt = "\n\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages)
    resp = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(temperature=temperature),
    )
    if not resp.text:
        raise HTTPException(status_code=502, detail="Gemini returned an empty response.")
    return resp.text



def get_llm_explanation(result: dict) -> str:
    return generate_gemini_text(
        [{"role": "user", "content": build_explanation_prompt(result)}],
        temperature=0.5,
    )



# ══════════════════════════════════════════════════════════════════════════════
#  6.  Pydantic models
# ══════════════════════════════════════════════════════════════════════════════
class ChatRequest(BaseModel):
    session_id: str
    message:    str

class NewSessionResponse(BaseModel):
    session_id: str

# ══════════════════════════════════════════════════════════════════════════════
#  7.  Routes
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/", response_class=HTMLResponse)
async def serve_ui(request: Request):
    """Serve the main chat interface."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/api/new-session")
async def new_session():
    """Create a new conversation session."""
    sid = str(uuid.uuid4())
    SESSIONS[sid] = []
    return {"session_id": sid}


@app.post("/api/chat")
async def chat(payload: ChatRequest):
    """
    Main chat endpoint.
    Returns:
      { type: "message",    html: "...", raw: "..." }
      { type: "prediction", churn: bool, prob: float, explanation_html: "...",
        profile: {...}, verdict: "...", badge: "..." }
    """
    sid = payload.session_id
    if sid not in SESSIONS:
        SESSIONS[sid] = []

    history = SESSIONS[sid]
    history.append({"role": "user", "content": payload.message})

    # ── Call Gemini ─────────────────────────────────────────────────────
 

    bot_text = generate_gemini_text(
        [{"role": "system", "content": SYSTEM_PROMPT}] + history,
        temperature=0.4,
    )
    history.append({"role": "assistant", "content": bot_text})

    # ── Check for JSON payload ────────────────────────────────────────────────
    extracted = extract_json_block(bot_text)
    if extracted:
        # Run ML model
        result = run_ml_prediction(extracted)

        # Get explanation from Gemini
        explanation_raw = get_llm_explanation(result)

        # Sanitise explanation with BeautifulSoup
        explanation_html = sanitise_html(explanation_raw)

        # Build profile table rows
        profile_rows = []
        for k, v in extracted.items():
            meta  = FEATURE_META.get(k, {})
            label = meta.get("label", k.strip())
            profile_rows.append({"field": label, "value": str(v)})

        return JSONResponse({
            "type":             "prediction",
            "churn":            result["churn"],
            "prob":             round(result["prob"] * 100, 1),
            "no_prob":          round(result["no_prob"] * 100, 1),
            "explanation_html": explanation_html,
            "profile":          profile_rows,
        })

    # ── Regular chat reply ────────────────────────────────────────────────────
    # Strip the JSON block if partially present, sanitise with BeautifulSoup
    display_text = re.sub(r"<JSON>.*?</JSON>", "", bot_text, flags=re.DOTALL).strip()
    safe_html    = sanitise_html(display_text)

    return JSONResponse({
        "type": "message",
        "html": safe_html,
        "raw":  display_text,
    })


@app.get("/api/health")
async def health():
    return {"status": "ok", "model": "RandomForest", "llm": GEMINI_MODEL}

