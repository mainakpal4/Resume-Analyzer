import os
import google.generativeai as genai
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel

# Load .env file
load_dotenv()

# Get API key
GENAI_API_KEY = os.getenv("GENAI_API_KEY")
if not GENAI_API_KEY:
    raise RuntimeError("GENAI_API_KEY not set in .env file")

# Configure Gemini
genai.configure(api_key=GENAI_API_KEY)

# Initialize FastAPI
app = FastAPI(title="Employee Sentiment Analysis API")

class FeedbackRequest(BaseModel):
    feedback: str

@app.get("/")
def home():
    return {"message": "Employee Sentiment Analysis API is running ✅"}

@app.post("/analyze")
def analyze(request: FeedbackRequest):
    feedback = request.feedback.strip()
    if not feedback:
        return {"error": "No feedback provided"}

    model = genai.GenerativeModel("gemini-2.5-flash")
    prompt = f"""
    Analyze the following employee feedback and return:
    1. Sentiment (Positive / Neutral / Negative)
    2. Attrition Risk (High / Medium / Low)
    3. Engagement Recommendations

    Feedback: {feedback}
    """
    response = model.generate_content(prompt)

    return {
        "feedback": feedback,
        "analysis": response.text
    }
