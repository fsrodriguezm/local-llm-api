from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import ollama

app = FastAPI(title="Local LLM API", description="Simple FastAPI with Ollama qwen2.5:3b")

class ChatRequest(BaseModel):
    prompt: str
    stream: bool = False
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Temperature for response randomness (0.0-2.0)")

class ChatResponse(BaseModel):
    response: str

@app.get("/")
async def root():
    return {"message": "Local LLM API is running", "model": "qwen2.5:3b"}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Send a prompt to the qwen2.5:3b model and get a response
    """
    try:
        response = ollama.chat(
            model="qwen2.5:3b",
            messages=[{"role": "user", "content": request.prompt}],
            stream=request.stream,
            options={"temperature": request.temperature}
        )
        
        if request.stream:
            # For streaming, we'll just return the full response
            # In a real streaming scenario, you'd use StreamingResponse
            full_response = ""
            for chunk in response:
                full_response += chunk['message']['content']
            return ChatResponse(response=full_response)
        else:
            return ChatResponse(response=response['message']['content'])
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error communicating with Ollama: {str(e)}")

@app.get("/health")
async def health():
    """
    Check if the API and Ollama are working
    """
    try:
        # Try to list models to verify Ollama is accessible
        models = ollama.list()
        return {"status": "healthy", "ollama_connected": True, "models_available": len(models.get('models', []))}
    except Exception as e:
        return {"status": "unhealthy", "ollama_connected": False, "error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
