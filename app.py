from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
from textSummarizer.pipeline.prediction import PredictionPipeline
from starlette.responses import RedirectResponse

# Initialize the FastAPI app
app = FastAPI(title="Samsum Text Summarizer API")

# Define the data format we expect the user to send
class ChatLog(BaseModel):
    text: str

# Redirect the base URL straight to the interactive web UI
@app.get("/")
def index():
    return RedirectResponse(url="/docs")

# The actual prediction endpoint
@app.post("/predict")
def predict_summary(chat: ChatLog):
    try:
        pipeline = PredictionPipeline()
        summary = pipeline.predict(chat.text)
        return {"summary": summary}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    # Run the local server on port 8000
    uvicorn.run(app, host="127.0.0.1", port=8000)