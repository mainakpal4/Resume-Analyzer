import os
import json
from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv
import google.generativeai as genai
from schemas import FeedbackRequest, FeedbackAnalysis  # Import schemas

# Load .env
load_dotenv()

GENAI_API_KEY = os.getenv("GENAI_API_KEY")
if not GENAI_API_KEY:
    raise RuntimeError("GENAI_API_KEY not set in .env file")

# Configure Gemini API
genai.configure(api_key=GENAI_API_KEY)

# Initialize FastAPI
app = FastAPI(title="Employee Sentiment Analysis API")

@app.get("/")
def home():
    return {"message": "Employee Sentiment Analysis API is running ✅"}

@app.post("/analyze", response_model=FeedbackAnalysis)
def analyze(request: FeedbackRequest):
    feedback = request.feedback

    model = genai.GenerativeModel("gemini-2.5-flash")

    # 🔒 Force strict JSON format
    prompt = f"""
    You are an HR analytics assistant. 
    Analyze the employee feedback below and return ONLY valid JSON (no text, no markdown, no explanation).
    The JSON must exactly follow this schema:
    {{
        "feedback": "<the original feedback>",
        "sentiment": "Positive | Neutral | Negative",
        "attrition_risk": "High | Medium | Low",
        "recommendations": ["<string>", "<string>", ...]
    }}

    Employee Feedback: "{feedback}"
    """

    response = model.generate_content(prompt)

    # Validate JSON response
    try:
        parsed = json.loads(response.text)

        # Ensure schema validation via Pydantic
        analysis = FeedbackAnalysis(
            feedback=parsed.get("feedback", feedback),
            sentiment=parsed.get("sentiment", "Unknown"),
            attrition_risk=parsed.get("attrition_risk", "Unknown"),
            recommendations=parsed.get("recommendations", []),
        )
        return analysis

    except (json.JSONDecodeError, ValueError):
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Invalid JSON response from Gemini",
                "raw_output": response.text,
                "expected_format": FeedbackAnalysis.model_json_schema()
            },
        )