from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import ollama
import sys

app = FastAPI(title="Local LLM API", description="FastAPI with Ollama - dynamic model selection")

SELECTED_MODEL: str = ""

class ChatRequest(BaseModel):
    prompt: str
    stream: bool = False
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Temperature for response randomness (0.0-2.0)")

class ChatResponse(BaseModel):
    response: str

def get_available_models() -> list[str]:
    """Fetch available models from Ollama."""
    try:
        result = ollama.list()
        return [model.model for model in result.models]
    except Exception as e:
        print(f"Error connecting to Ollama: {e}")
        return []

def select_model() -> str:
    """CLI prompt to select a model."""
    models = get_available_models()

    if not models:
        print("No models found. Make sure Ollama is running and has models installed.")
        print("Install a model with: ollama pull <model-name>")
        sys.exit(1)

    print("\nAvailable Ollama models:")
    print("-" * 30)
    for i, model in enumerate(models, 1):
        print(f"  {i}. {model}")
    print("-" * 30)

    while True:
        try:
            choice = input(f"Select a model (1-{len(models)}): ").strip()
            index = int(choice) - 1
            if 0 <= index < len(models):
                selected = models[index]
                print(f"Selected: {selected}\n")
                return selected
            else:
                print(f"Please enter a number between 1 and {len(models)}")
        except ValueError:
            print("Please enter a valid number")
        except KeyboardInterrupt:
            print("\nExiting...")
            sys.exit(0)

@app.get("/")
async def root():
    return {"message": "Local LLM API is running", "model": SELECTED_MODEL}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Send a prompt to the selected model and get a response
    """
    try:
        response = ollama.chat(
            model=SELECTED_MODEL,
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
        models = ollama.list()
        return {
            "status": "healthy",
            "ollama_connected": True,
            "selected_model": SELECTED_MODEL,
            "models_available": len(models.models)
        }
    except Exception as e:
        return {"status": "unhealthy", "ollama_connected": False, "error": str(e)}

if __name__ == "__main__":
    import uvicorn
    SELECTED_MODEL = select_model()
    print(f"Starting API server with model: {SELECTED_MODEL}")
    uvicorn.run(app, host="0.0.0.0", port=8000)
