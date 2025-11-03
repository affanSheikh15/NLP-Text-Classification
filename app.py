from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import pipeline
import uvicorn
from typing import List, Dict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Sentiment Analysis API",
    description="Real-time sentiment analysis using DistilBERT",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the sentiment analysis model
logger.info("Loading DistilBERT model...")
try:
    sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        device=-1  # Use CPU; change to 0 for GPU
    )
    logger.info("Model loaded successfully!")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    sentiment_pipeline = None

# Request/Response models
class TextInput(BaseModel):
    text: str
    
from typing import List, Dict, Union

class SentimentOutput(BaseModel):
    text: str
    sentiment: str
    confidence: float
    all_scores: List[Dict[str, Union[str, float]]]


class BatchTextInput(BaseModel):
    texts: List[str]

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Sentiment Analysis API",
        "model": "distilbert-base-uncased-finetuned-sst-2-english",
        "endpoints": {
            "/analyze": "POST - Analyze single text",
            "/batch-analyze": "POST - Analyze multiple texts",
            "/health": "GET - Check API health"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": sentiment_pipeline is not None
    }

@app.post("/analyze", response_model=SentimentOutput)
async def analyze_sentiment(input_data: TextInput):
    """
    Analyze sentiment of a single text input
    
    Args:
        input_data: TextInput object containing the text to analyze
        
    Returns:
        SentimentOutput with sentiment label and confidence score
    """
    if sentiment_pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not input_data.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    try:
        # Get prediction
        result = sentiment_pipeline(input_data.text[:512])[0]  # Limit to 512 chars
        
        # Map label to our categories
        sentiment = result['label']
        confidence = result['score']
        
        # Determine neutral threshold
        if confidence < 0.6:
            sentiment = "NEUTRAL"
        
        return SentimentOutput(
            text=input_data.text,
            sentiment=sentiment,
            confidence=round(confidence, 4),
            all_scores=[{
                "label": result['label'],
                "score": round(result['score'], 4)
            }]
        )
    except Exception as e:
        logger.error(f"Error analyzing sentiment: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch-analyze")
async def batch_analyze_sentiment(input_data: BatchTextInput):
    """
    Analyze sentiment of multiple texts
    
    Args:
        input_data: BatchTextInput object containing list of texts
        
    Returns:
        List of sentiment results
    """
    if sentiment_pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not input_data.texts:
        raise HTTPException(status_code=400, detail="Texts list cannot be empty")
    
    try:
        results = []
        for text in input_data.texts[:10]:  # Limit to 10 texts per batch
            if text.strip():
                result = sentiment_pipeline(text[:512])[0]
                sentiment = result['label']
                confidence = result['score']
                
                if confidence < 0.6:
                    sentiment = "NEUTRAL"
                
                results.append({
                    "text": text,
                    "sentiment": sentiment,
                    "confidence": round(confidence, 4)
                })
        
        return {"results": results}
    except Exception as e:
        logger.error(f"Error in batch analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
import gradio as gr
import threading

# --- Interactive Gradio UI Function ---
def gradio_sentiment(text):
    if not text.strip():
        return {"Error": "Text cannot be empty."}
    result = sentiment_pipeline(text[:512])[0]
    sentiment = result["label"]
    confidence = result["score"]
    if confidence < 0.6:
        sentiment = "NEUTRAL"
    return {"Sentiment": sentiment, "Confidence": round(confidence, 4)}

# --- Custom Interactive CSS ---
custom_css = """
body {
    background: radial-gradient(circle at top left, #0f2027, #203a43, #2c5364);
    font-family: 'Inter', sans-serif;
    color: #e0e0e0;
}

h1 {
    text-align: center;
    color: #00bcd4;
    animation: fadeSlideIn 1.5s ease;
}

textarea {
    background-color: #1e293b !important;
    color: #e0e0e0 !important;
    border-radius: 12px !important;
    border: 1px solid #334155 !important;
    transition: box-shadow 0.3s ease;
}
textarea:focus {
    box-shadow: 0 0 10px #00bcd4 !important;
}

button {
    background: linear-gradient(135deg, #00bcd4, #0288d1) !important;
    color: white !important;
    border-radius: 10px !important;
    font-weight: bold !important;
    transition: all 0.3s ease !important;
}
button:hover {
    transform: translateY(-3px);
    box-shadow: 0 4px 15px rgba(0, 188, 212, 0.4);
}

.gradio-container {
    box-shadow: 0 0 20px rgba(0, 188, 212, 0.2);
    border-radius: 20px;
    padding: 20px;
    background: rgba(255, 255, 255, 0.05);
}

@keyframes fadeSlideIn {
    from { opacity: 0; transform: translateY(-20px); }
    to { opacity: 1; transform: translateY(0); }
}
"""

# --- Create Interactive Gradio Interface ---
with gr.Blocks(css=custom_css, title="Interactive Sentiment App") as interface:
    gr.Markdown("# 💬 Sentiment Analysis (Interactive UI)")
    gr.Markdown("Experience real-time text emotion detection with a smooth, modern interface.")
    text_input = gr.Textbox(label="Type your text here...", placeholder="e.g. I love using this app!", lines=4)
    output = gr.JSON(label="Sentiment Result")
    analyze_button = gr.Button("Analyze Sentiment")
    analyze_button.click(gradio_sentiment, inputs=text_input, outputs=output)

# --- Launch Gradio in background thread ---
def launch_gradio():
    interface.launch(server_name="0.0.0.0", server_port=7860, share=True)

threading.Thread(target=launch_gradio, daemon=True).start()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
