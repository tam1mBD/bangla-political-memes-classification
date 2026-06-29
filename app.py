import os
import shutil
import tempfile
import typing
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import torch
import numpy as np
from PIL import Image

from memeClassifier import logger
from memeClassifier.pipeline.meme_prediction import MemePredictionPipeline


# Modern Lifespan handler (Fixes the DeprecationWarning & caches pipeline state safely)
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing prediction pipeline components inside FastAPI server context...")
    # Cache heavy multi-modal ensemble components into global application state
    app.state.pipeline = MemePredictionPipeline()
    logger.info("✓ Application Pipeline loaded and ready for serving requests.")
    yield
    logger.info("Shutting down lifespan context, releasing pipeline resources.")


app = FastAPI(
    title="Bangla Political Meme Classifier",
    description="Multi-Modal Stacking Ensemble Pipeline using OCR, Text Models, and Vision CLIP",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for cross-origin compliance
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def read_index():
    """Serves the main dashboard if index.html exists, otherwise falls back to an API Health info screen."""
    index_path = os.path.join(os.path.dirname(__file__), "templates/index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    
    return JSONResponse(
        status_code=200,
        content={
            "status": "healthy",
            "message": "Bangla Political Meme Classifier Pipeline is fully operational!",
            "ui_notice": "index.html was not found in this root folder.",
            "interactive_testing_dashboard": "http://127.0.0.1:8000/docs"
        }
    )


@app.post("/predict")
async def predict_meme(request: Request, file: UploadFile = File(...)):
    """
    Accepts an uploaded meme image, processes text/vision parameters,
    and forwards array metrics to the meta stacking classifier.
    """
    try:
        # FIX: Point securely to request.app.state instead of request.state
        pipeline: MemePredictionPipeline = request.app.state.pipeline
    except AttributeError:
        raise HTTPException(status_code=503, detail="Prediction model pipeline is uninitialized.")

    if pipeline is None:
        raise HTTPException(status_code=503, detail="Prediction model pipeline is uninitialized.")

    # Explicitly isolate and validate the filename to satisfy static type checkers
    filename = file.filename
    if not filename:
        raise HTTPException(status_code=400, detail="Uploaded file is missing a valid filename.")

    if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp')):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image file.")

    try:
        # Save upload to a secure isolated temporary path to prevent thread cross-contamination
        suffix = os.path.splitext(filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name

        logger.info(f"Received web upload target saved temporarily to: {tmp_path}")

        # Execute classification run via pipeline logic
        result = pipeline.ocr_reader.readtext(tmp_path)
        detected_texts = [
            detection[1] for detection in result 
            if isinstance(detection, (tuple, list)) and len(detection) >= 2 and isinstance(detection[1], str)
        ]
        raw_text = ' '.join(detected_texts)
        processed_text = pipeline.preprocess_text(raw_text)

        # Build feature count dimensions
        words = processed_text.lower().split()
        word_count = len(words)
        matches = sum(1 for word in words if word in pipeline.political_words)
        ratio = matches / word_count if word_count > 0 else 0.0
        text_features = [[matches, ratio]]

        # Base Probability checks
        lr_p = pipeline.lr_model.predict_proba(text_features)[0] if pipeline.lr_model else [0.5, 0.5]
        
        X_tensor = torch.FloatTensor(text_features).to(pipeline.device)
        with torch.no_grad():
            nn_out = pipeline.nn_model(X_tensor).cpu().numpy()[0][0]
        nn_p = [float(1.0 - nn_out), float(nn_out)]

        img = Image.open(tmp_path).convert('RGB')
        
        # Cast processor to Any to bypass static type-stub errors regarding dynamic parameters
        processor_any: typing.Any = pipeline.clip_processor
        clip_inputs = processor_any(images=img, return_tensors="pt")
        
        pixel_values = clip_inputs['pixel_values'].to(pipeline.device)
        with torch.no_grad():
            clip_logits = pipeline.clip_model(pixel_values)
            clip_p = torch.softmax(clip_logits, dim=1).cpu().numpy()[0]

        # Calculate Ensemble predictions
        stacking_vector = pipeline.create_stacking_features(np.array([lr_p]), np.array([nn_p]), np.array([clip_p]))
        final_prediction = pipeline.meta_model.predict(stacking_vector)[0]
        final_proba = pipeline.meta_model.predict_proba(stacking_vector)[0]

        label_mapping = {0: "NonPolitical", 1: "Political"}
        predicted_label = label_mapping[final_prediction]
        confidence = float(final_proba[final_prediction])

        # Cleanup temporary file from storage
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

        # Build structured json output payload
        return {
            "status": "success",
            "extracted_text": raw_text if raw_text.strip() != "" else "[No Text Found via OCR]",
            "cleaned_text": processed_text if processed_text.strip() != "" else "[No usable tokens extracted]",
            "metrics": {
                "keyword_matches": matches,
                "density_ratio": round(ratio, 4)
            },
            "prediction": {
                "label": predicted_label,
                "confidence_percentage": round(confidence * 100, 2)
            },
            "base_models_breakdown": {
                "logistic_regression": {"non_political": round(float(lr_p[0]), 4), "political": round(float(lr_p[1]), 4)},
                "deep_neural_network": {"non_political": round(float(nn_p[0]), 4), "political": round(float(nn_p[1]), 4)},
                "vision_clip_model": {"non_political": round(float(clip_p[0]), 4), "political": round(float(clip_p[1]), 4)}
            }
        }

    except Exception as e:
        logger.exception(f"Exception encountered during active web handler run: {e}")
        raise HTTPException(status_code=500, detail=f"Inference Pipeline Failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)