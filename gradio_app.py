import gradio as gr
from transformers import pipeline
import time

# Load the sentiment analysis model
print("Loading DistilBERT model...")
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    device=-1
)
print("Model loaded successfully!")

def analyze_sentiment(text):
    """
    Analyze sentiment of input text
    
    Args:
        text: Input text string
        
    Returns:
        Tuple of (sentiment, confidence, html_output)
    """
    if not text.strip():
        return "⚠️ NEUTRAL", 0.0, "<p style='color: #666;'>Please enter some text to analyze.</p>"
    
    try:
        # Add slight delay for animation effect
        time.sleep(0.3)
        
        # Get prediction
        result = sentiment_pipeline(text[:512])[0]
        
        sentiment = result['label']
        confidence = result['score']
        
        # Determine sentiment with neutral threshold
        if confidence < 0.6:
            sentiment = "NEUTRAL"
            emoji = "😐"
            color = "#FFA500"
        elif sentiment == "POSITIVE":
            emoji = "😊"
            color = "#4CAF50"
        else:  # NEGATIVE
            emoji = "😞"
            color = "#F44336"
            sentiment = "NEGATIVE"
        
        # Create detailed HTML output
        confidence_percent = confidence * 100
        
        html_output = f"""
        <div style='padding: 20px; border-radius: 10px; background: linear-gradient(135deg, {color}22 0%, {color}11 100%); border-left: 4px solid {color};'>
            <h2 style='color: {color}; margin: 0 0 15px 0;'>
                {emoji} Sentiment: {sentiment}
            </h2>
            <div style='margin: 15px 0;'>
                <p style='margin: 5px 0; font-size: 16px;'><strong>Confidence Score:</strong> {confidence_percent:.2f}%</p>
                <div style='background: #eee; border-radius: 10px; height: 25px; overflow: hidden; margin: 10px 0;'>
                    <div style='background: {color}; height: 100%; width: {confidence_percent}%; border-radius: 10px; transition: width 0.5s ease;'></div>
                </div>
            </div>
            <div style='margin-top: 20px; padding: 15px; background: white; border-radius: 8px;'>
                <p style='margin: 0; color: #555; font-style: italic;'>"{text[:100]}{'...' if len(text) > 100 else ''}"</p>
            </div>
        </div>
        """
        
        return f"{emoji} {sentiment}", f"{confidence_percent:.2f}%", html_output
        
    except Exception as e:
        return "❌ ERROR", "0.00%", f"<p style='color: red;'>Error: {str(e)}</p>"

# Example texts
examples = [
    ["I absolutely love this product! It exceeded all my expectations and the quality is outstanding."],
    ["This is the worst experience I've ever had. Completely disappointed and frustrated."],
    ["The movie was okay. Nothing special but not terrible either."],
    ["Thank you so much for your help! You made my day!"],
    ["I'm very unhappy with the service. This is unacceptable."]
]

# Custom CSS for animations and styling
custom_css = """
#sentiment-output {
    font-size: 24px !important;
    font-weight: bold !important;
    text-align: center !important;
    padding: 15px !important;
    border-radius: 10px !important;
    animation: fadeIn 0.5s ease-in !important;
}

#confidence-output {
    font-size: 20px !important;
    font-weight: bold !important;
    text-align: center !important;
    color: #2196F3 !important;
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(-10px); }
    to { opacity: 1; transform: translateY(0); }
}

.gradio-container {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
}

#analyze-btn {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    border: none !important;
    font-size: 18px !important;
    padding: 12px 30px !important;
    transition: transform 0.2s ease !important;
}

#analyze-btn:hover {
    transform: scale(1.05) !important;
}

.prose p {
    margin: 0 !important;
}
"""

# Create Gradio interface
with gr.Blocks(css=custom_css, theme=gr.themes.Soft(primary_hue="purple")) as demo:
    gr.Markdown(
        """
        # 🎭 Sentiment Analysis with DistilBERT
        ### Analyze the emotional tone of your text in real-time
        
        Powered by **DistilBERT** - a fast and accurate transformer model for sentiment classification.
        """
    )
    
    with gr.Row():
        with gr.Column(scale=2):
            text_input = gr.Textbox(
                label="📝 Enter your text here",
                placeholder="Type or paste your text to analyze its sentiment...",
                lines=5,
                max_lines=10
            )
            
            analyze_btn = gr.Button("🔍 Analyze Sentiment", variant="primary", elem_id="analyze-btn")
            
            gr.Examples(
                examples=examples,
                inputs=text_input,
                label="💡 Try these examples"
            )
        
        with gr.Column(scale=1):
            sentiment_output = gr.Textbox(
                label="🎯 Sentiment Result",
                interactive=False,
                elem_id="sentiment-output"
            )
            confidence_output = gr.Textbox(
                label="📊 Confidence",
                interactive=False,
                elem_id="confidence-output"
            )
    
    with gr.Row():
        html_output = gr.HTML(label="📈 Detailed Analysis")
    
    gr.Markdown(
        """
        ---
        ### ℹ️ About
        - **Model**: DistilBERT (distilbert-base-uncased-finetuned-sst-2-english)
        - **Categories**: Positive, Negative, Neutral (low confidence)
        - **Max Length**: 512 characters
        
        ### 🚀 API Endpoints
        - `POST /analyze` - Single text analysis
        - `POST /batch-analyze` - Multiple texts analysis
        - `GET /health` - Health check
        """
    )
    
    # Set up the event handler
    analyze_btn.click(
        fn=analyze_sentiment,
        inputs=text_input,
        outputs=[sentiment_output, confidence_output, html_output]
    )
    
    # Also trigger on Enter key
    text_input.submit(
        fn=analyze_sentiment,
        inputs=text_input,
        outputs=[sentiment_output, confidence_output, html_output]
    )

# Launch the app
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )
