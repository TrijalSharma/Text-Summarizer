from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
import gradio as gr
from textSummarizer.pipeline.prediction import PredictionPipeline

# Initialize the pipeline and API
pipeline = PredictionPipeline()
app = FastAPI(title="Samsum Text Summarizer API")

class ChatLog(BaseModel):
    text: str

# Keep your enterprise API route hidden in the background
@app.post("/predict")
def predict_summary(chat: ChatLog):
    return {"summary": pipeline.predict(chat.text)}

# Create the visual Gradio interface
def summarize_for_ui(text):
    return pipeline.predict(text)

demo = gr.Interface(
    fn=summarize_for_ui,
    inputs=gr.Textbox(lines=10, placeholder="Paste a messy chat log here..."),
    outputs=gr.Textbox(label="AI Summary"),
    title="Pegasus Text Summarizer",
    description="Custom-trained model to compress text message dialogues.",
    theme="soft"
)

# Mount the beautiful UI onto your existing FastAPI server
app = gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)