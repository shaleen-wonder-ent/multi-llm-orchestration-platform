

import os
import logging
import traceback
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from openai import AzureOpenAI, OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

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
        "PHI4_ENDPOINT": bool(os.getenv("PHI4_ENDPOINT")),
        "DEEPSEEK_ENDPOINT": bool(os.getenv("DEEPSEEK_ENDPOINT"))
    }}


def query_llm(model_name: str, prompt: str) -> str:
    logger.info(f"Starting API call for model: {model_name}")
    logger.info(f"Prompt: {prompt[:100]}..." if len(prompt) > 100 else f"Prompt: {prompt}")
    
    if model_name.lower() == "gpt-5.2-chat":
        endpoint = os.getenv("GPT52_ENDPOINT")
        key = os.getenv("GPT52_KEY")
        deployment = os.getenv("GPT52_DEPLOYMENT", "gpt-5.2-chat")
        api_version = os.getenv("GPT52_API_VERSION", "2025-04-01-preview")
        
        logger.info(f"GPT-5.2 Config - Endpoint: {endpoint}, Deployment: {deployment}, API Version: {api_version}")
        
        try:
            client = AzureOpenAI(
                api_version=api_version,
                azure_endpoint=endpoint,
                api_key=key,
            )
            response = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=1024,
                model=deployment
            )
            content = response.choices[0].message.content
            if content is None:
                return "GPT-5.2-Chat returned empty response (None)"
            stripped_content = content.strip()
            return stripped_content if stripped_content else "GPT-5.2-Chat returned empty content"
            
        except Exception as e:
            logger.error(f"GPT-5.2 API error: {type(e).__name__}: {str(e)}")
            return f"GPT-5.2-Chat API error: {type(e).__name__}: {str(e)}"

    elif model_name.lower() == "phi-4-mini-reasoning":
        endpoint = os.getenv("PHI4_ENDPOINT")
        key = os.getenv("PHI4_KEY")
        deployment = os.getenv("PHI4_DEPLOYMENT", "Phi-4-mini-reasoning")
        api_version = os.getenv("PHI4_API_VERSION", "2024-05-01-preview")
        
        logger.info(f"PHI-4 Config - Endpoint: {endpoint}, Deployment: {deployment}")
        
        try:
            # Azure AI model inference (serverless) uses OpenAI client with base_url
            client = OpenAI(
                base_url=endpoint,
                api_key=key,
            )
            response = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=512,
                model=deployment
            )
            content = response.choices[0].message.content
            if content is None:
                return "Phi-4-mini-reasoning returned empty response (None)"
            stripped_content = content.strip()
            return stripped_content if stripped_content else "Phi-4-mini-reasoning returned empty content"
                
        except Exception as e:
            logger.error(f"PHI-4 API error: {type(e).__name__}: {str(e)}")
            return f"Phi-4-mini-reasoning API error: {type(e).__name__}: {str(e)}"

    elif model_name.lower() == "deepseek-v3.2":
        endpoint = os.getenv("DEEPSEEK_ENDPOINT")
        key = os.getenv("DEEPSEEK_KEY")
        deployment = os.getenv("DEEPSEEK_DEPLOYMENT", "DeepSeek-V3.2")
        api_version = os.getenv("DEEPSEEK_API_VERSION", "2024-05-01-preview")
        
        logger.info(f"DeepSeek Config - Endpoint: {endpoint}, Deployment: {deployment}")
        
        try:
            client = OpenAI(
                base_url=endpoint,
                api_key=key,
            )
            response = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1024,
                model=deployment
            )
            content = response.choices[0].message.content
            if content is None:
                return "DeepSeek-V3.2 returned empty response (None)"
            stripped_content = content.strip()
            return stripped_content if stripped_content else "DeepSeek-V3.2 returned empty content"
                
        except Exception as e:
            logger.error(f"DeepSeek API error: {type(e).__name__}: {str(e)}")
            return f"DeepSeek-V3.2 API error: {type(e).__name__}: {str(e)}"

    else:
        return f"Model {model_name} not supported."

import json

def triage_responses(prompt: str, responses: dict) -> tuple:
    """Use GPT-5.2 as a judge to evaluate and grade all model responses."""
    logger.info("Starting LLM-as-Judge evaluation")
    
    # Build the judging prompt
    responses_text = ""
    for model, response in responses.items():
        # Truncate very long responses for the judge
        truncated = response[:2000] if len(response) > 2000 else response
        responses_text += f"\n--- {model} ---\n{truncated}\n"
    
    judge_prompt = f"""You are an expert judge evaluating AI model responses. 
The user asked: "{prompt}"

Here are the responses from different models:
{responses_text}

Evaluate each response on a scale of 1-10 based on:
- Accuracy and correctness
- Completeness and depth
- Clarity and readability
- Relevance to the question

Return your evaluation as valid JSON only (no markdown, no code fences), in this exact format:
{{
  "evaluations": {{
    "<model_name>": {{
      "grade": <number 1-10>,
      "reason": "<1-2 sentence explanation of the grade>"
    }}
  }},
  "best": "<model_name of the best response>",
  "summary": "<1 sentence overall summary of why the best was chosen>"
}}"""

    try:
        endpoint = os.getenv("GPT52_ENDPOINT")
        key = os.getenv("GPT52_KEY")
        deployment = os.getenv("GPT52_DEPLOYMENT", "gpt-5.2-chat")
        api_version = os.getenv("GPT52_API_VERSION", "2025-04-01-preview")
        
        client = AzureOpenAI(
            api_version=api_version,
            azure_endpoint=endpoint,
            api_key=key,
        )
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": judge_prompt}],
            max_completion_tokens=1024,
            model=deployment
        )
        content = response.choices[0].message.content
        if content:
            content = content.strip()
            # Remove markdown code fences if present
            if content.startswith("```"):
                content = content.split("\n", 1)[1] if "\n" in content else content[3:]
                content = content.rsplit("```", 1)[0]
                content = content.strip()
            
            judge_result = json.loads(content)
            evaluations = judge_result.get("evaluations", {})
            grades = {model: eval_data["grade"] for model, eval_data in evaluations.items()}
            reasons = {model: eval_data["reason"] for model, eval_data in evaluations.items()}
            best_model = judge_result.get("best", max(grades, key=lambda k: grades[k]))
            summary = judge_result.get("summary", "")
            
            logger.info(f"Judge result: grades={grades}, best={best_model}")
            return best_model, grades, reasons, summary
    except Exception as e:
        logger.error(f"LLM Judge error: {type(e).__name__}: {str(e)}")
    
    # Fallback to simple heuristic if judge fails
    logger.warning("Falling back to heuristic grading")
    grades = {}
    reasons = {}
    for model, resp in responses.items():
        if "error" in resp.lower():
            grades[model] = 1
            reasons[model] = "Response contains an error."
        else:
            length_score = min(len(resp) / 100, 5)
            content_score = 3 if any(w in resp.lower() for w in ['because', 'therefore', 'however', 'specifically']) else 1
            grades[model] = min(length_score + content_score, 10)
            reasons[model] = "Scored by heuristic (LLM judge unavailable)."
    best_model = max(grades, key=lambda k: grades[k])
    return best_model, grades, reasons, "Graded by heuristic fallback."

@app.post("/ask")
async def ask(request: AskRequest):
    prompt = request.query or request.prompt
    if not prompt:
        logger.warning("Empty prompt received")
        return {"error": "Please provide either 'prompt' or 'query' field"}
    
    logger.info(f"Processing request with prompt: {prompt[:100]}..." if len(prompt) > 100 else f"Processing request with prompt: {prompt}")
    
    models = ["gpt-5.2-chat", "phi-4-mini-reasoning", "deepseek-v3.2"]
    responses = {}
    
    for model in models:
        logger.info(f"Calling model: {model}")
        response = query_llm(model, prompt)
        responses[model] = response
        logger.info(f"Model {model} response length: {len(response)}")
    
    logger.info(f"All responses collected")
    best_model, grades, reasons, judge_summary = triage_responses(prompt, responses)
    
    result = {
        "prompt": prompt,
        "answers": responses,
        "best": best_model,
        "grades": grades,
        "reasons": reasons,
        "judge_summary": judge_summary,
        "judge": "gpt-5.2-chat",
        "responses": responses,
        "best_answer": responses[best_model]
    }
    
    logger.info(f"Final result being returned: {result}")
    return result

# Add /api/llm endpoint for compatibility
@app.post("/api/llm")
async def api_llm(request: AskRequest):
    logger.info("API /api/llm endpoint called")
    return await ask(request)

# TODO: Add tracing/debugging instrumentation
# TODO: Integrate with agentdev and VSCode debugging

# Mount static files at root path to serve React assets
app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8087, reload=True)
