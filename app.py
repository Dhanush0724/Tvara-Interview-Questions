"""
Gemini 2.0 Flash API Server
A minimal FastAPI server that forwards prompts to the Gemini 2.0 Flash API.
"""

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
import requests
from typing import Optional, Dict, Any

API_KEY = "AIzaSyDiET0PoOJSlWDJELnbi7JTurC5GlHrN64"
ENDPOINT = (
    "https://generativelanguage.googleapis.com/v1beta/"
    "models/gemini-2.0-flash:generateContent"
)

app = FastAPI(
    title="Gemini 2.0 Flash API",
    description="Simple API wrapper for Google's Gemini 2.0 Flash model",
    version="1.0.0"
)


class GenerateRequest(BaseModel):
    """Request model for text generation."""
    prompt: str = Field(..., description="The prompt to send to Gemini", min_length=1)
    debug: bool = Field(False, description="Include raw API response in output")


class GenerateResponse(BaseModel):
    """Response model for text generation."""
    output: str = Field(..., description="The generated text from Gemini")
    raw: Optional[Dict[Any, Any]] = Field(None, description="Raw API response (only if debug=true)")


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest) -> GenerateResponse:
    """
    Generate text using Gemini 2.0 Flash.
    
    Args:
        request: The generation request containing the prompt and debug flag
        
    Returns:
        The generated text and optionally the raw API response
        
    Raises:
        HTTPException: If the API request fails
    """
    payload = {
        "contents": [
            {
                "parts": [{"text": request.prompt}]
            }
        ]
    }

    try:
        response = requests.post(
            f"{ENDPOINT}?key={API_KEY}",
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=30
        )
        response.raise_for_status()
        data = response.json()

        # Extract generated text
        generated_text = data["candidates"][0]["content"]["parts"][0]["text"]

        # Return response with or without raw data
        if request.debug:
            return GenerateResponse(output=generated_text, raw=data)
        else:
            return GenerateResponse(output=generated_text)

    except requests.exceptions.HTTPError as e:
        raise HTTPException(
            status_code=e.response.status_code if e.response else 500,
            detail=f"Gemini API error: {str(e)}"
        )
    except KeyError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected API response format: {str(e)}"
        )
    except requests.exceptions.RequestException as e:
        raise HTTPException(
            status_code=500,
            detail=f"Request failed: {str(e)}"
        )


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Gemini 2.0 Flash API",
        "version": "1.0.0",
        "endpoints": {
            "/generate": "POST - Generate text from a prompt",
            "/docs": "GET - Interactive API documentation"
        }
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)