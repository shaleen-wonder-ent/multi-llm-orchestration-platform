

import os
import logging
import traceback
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from openai import AzureOpenAI
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Pydantic model for request body
class AskRequest(BaseModel):
    prompt: str = None
    query: str = None


app = FastAPI()

# Serve React app for root route first
@app.get("/")
def read_index():
    return FileResponse("static/index.html")

# API status endpoint
@app.get("/status")
def api_status():
    return {"message": "Multi-LLM Orchestration API", "status": "running", "endpoints": ["/ask", "/api/llm"]}

# Simple test endpoint
@app.get("/test")
def test_endpoint():
    return {"status": "OK", "message": "Test endpoint working"}

# Health check endpoint
@app.post("/health") 
def health_check():
    return {"status": "healthy", "environment_variables": {
        "GPT52_ENDPOINT": bool(os.getenv("GPT52_ENDPOINT")),
        "PHI4_ENDPOINT": bool(os.getenv("PHI4_ENDPOINT"))
    }}

# Load environment variables
load_dotenv()


def query_llm(model_name: str, prompt: str) -> str:
    logger.info(f"Starting API call for model: {model_name}")
    logger.info(f"Prompt: {prompt[:100]}..." if len(prompt) > 100 else f"Prompt: {prompt}")
    
    if model_name.lower() == "gpt-5.2-chat":
        endpoint = os.getenv("GPT52_ENDPOINT")
        key = os.getenv("GPT52_KEY")
        deployment = os.getenv("GPT52_DEPLOYMENT")
        api_version = os.getenv("GPT52_API_VERSION")
        
        logger.info(f"GPT-5.2 Config - Endpoint: {endpoint}, Deployment: {deployment}, API Version: {api_version}")
        logger.info(f"GPT-5.2 Key present: {bool(key)}")
        
        try:
            client = AzureOpenAI(
                api_version=api_version,
                azure_endpoint=endpoint,
                api_key=key,
            )
            logger.info("GPT-5.2 client created successfully")
            
            response = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=1024,  # Increased to allow reasoning + visible output
                model=deployment
            )
            logger.info(f"GPT-5.2 response received: {response}")
            
            content = response.choices[0].message.content
            logger.info(f"GPT-5.2 content: '{content}'")
            logger.info(f"GPT-5.2 content length: {len(content) if content else 0}")
            
            if content is None:
                logger.warning("GPT-5.2 returned None content")
                return "GPT-5.2-Chat returned empty response (None)"
            
            stripped_content = content.strip()
            logger.info(f"GPT-5.2 stripped content length: {len(stripped_content)}")
            
            return stripped_content if stripped_content else "GPT-5.2-Chat returned empty content"
            
        except Exception as e:
            logger.error(f"GPT-5.2 API error: {type(e).__name__}: {str(e)}")
            logger.error(f"GPT-5.2 API error traceback: {traceback.format_exc()}")
            return f"GPT-5.2-Chat API error: {type(e).__name__}: {str(e)}"
    elif model_name.lower() == "phi-4-mini-reasoning":
        endpoint = os.getenv("PHI4_ENDPOINT")
        key = os.getenv("PHI4_KEY")
        deployment = os.getenv("PHI4_DEPLOYMENT")
        api_version = os.getenv("PHI4_API_VERSION")
        
        logger.info(f"PHI-4 Config - Endpoint: {endpoint}, Deployment: {deployment}, API Version: {api_version}")
        logger.info(f"PHI-4 Key present: {bool(key)}")
        
        try:
            client = AzureOpenAI(
                api_version=api_version,
                azure_endpoint=endpoint,
                api_key=key,
            )
            logger.info("PHI-4 client created successfully")
            
            response = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=512,  # Standardized parameter name
                model=deployment
            )
            logger.info(f"PHI-4 response received: {response}")
            
            content = response.choices[0].message.content
            logger.info(f"PHI-4 content: '{content}'")
            logger.info(f"PHI-4 content length: {len(content) if content else 0}")
            
            if content is None:
                logger.warning("PHI-4 returned None content")
                return "Phi-4-mini-reasoning returned empty response (None)"
                
            stripped_content = content.strip()
            logger.info(f"PHI-4 stripped content length: {len(stripped_content)}")
            
            return stripped_content if stripped_content else "Phi-4-mini-reasoning returned empty content"
            
        except Exception as e:
            logger.error(f"PHI-4 API error: {type(e).__name__}: {str(e)}")
            logger.error(f"PHI-4 API error traceback: {traceback.format_exc()}")
            return f"Phi-4-mini-reasoning API error: {type(e).__name__}: {str(e)}"
    else:
        return f"Model {model_name} not supported."

def triage_responses(responses: dict) -> str:
    logger.info(f"Starting triage with responses: {responses}")
    # Simple grading - count tokens as a basic quality metric
    grades = {}
    for model, response in responses.items():
        logger.info(f"Grading model {model} with response: '{response}'")
        if "error" in response.lower():
            grades[model] = 1
            logger.info(f"Model {model} has error, grade: 1")
        else:
            # Basic scoring based on response length and content quality
            length_score = min(len(response) / 100, 5)  # Up to 5 points for length
            content_score = 3 if any(word in response.lower() for word in ['because', 'therefore', 'however', 'specifically']) else 1
            grades[model] = min(length_score + content_score, 10)
            logger.info(f"Model {model} scores - length: {length_score}, content: {content_score}, total: {grades[model]}")
    
    logger.info(f"All grades: {grades}")
    # Return model with highest grade
    best_model = max(grades.keys(), key=lambda k: grades[k])
    logger.info(f"Best model selected: {best_model} with grade: {grades[best_model]}")
    return best_model, grades

@app.post("/ask")
async def ask(request: AskRequest):
    prompt = request.query or request.prompt
    if not prompt:
        logger.warning("Empty prompt received")
        return {"error": "Please provide either 'prompt' or 'query' field"}
    
    logger.info(f"Processing request with prompt: {prompt[:100]}..." if len(prompt) > 100 else f"Processing request with prompt: {prompt}")
    
    models = ["gpt-5.2-chat", "phi-4-mini-reasoning"]
    responses = {}
    
    for model in models:
        logger.info(f"Calling model: {model}")
        response = query_llm(model, prompt)
        responses[model] = response
        logger.info(f"Model {model} response length: {len(response)}")
    
    logger.info(f"All responses collected: {responses}")
    best_model, grades = triage_responses(responses)
    
    result = {
        "prompt": prompt,
        "answers": responses,  # Frontend expects 'answers' 
        "best": best_model,   # Frontend expects 'best'
        "grades": grades,     # Frontend expects 'grades'
        "responses": responses,  # Keep for backward compatibility
        "best_answer": responses[best_model]  # Keep for backward compatibility
    }
    
    logger.info(f"Final result being returned: {result}")
    return result

# Add /api/llm endpoint for compatibility
@app.post("/api/llm")
async def api_llm(request: AskRequest):
    logger.info("API /api/llm endpoint called")
    return await ask(request)

# Test endpoints
@app.get("/test")
def test_endpoint():
    return {"status": "OK", "message": "Test endpoint working"}

@app.post("/health")
def health_check():
    return {
        "status": "healthy",
        "environment_variables": {
            "GPT52_ENDPOINT": bool(os.getenv("GPT52_ENDPOINT")),
            "PHI4_ENDPOINT": bool(os.getenv("PHI4_ENDPOINT"))
        }
    }

# TODO: Add tracing/debugging instrumentation
# TODO: Integrate with agentdev and VSCode debugging

# Mount static files at root path to serve React assets
app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8087, reload=True)
