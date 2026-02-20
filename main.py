

import os
import logging
import traceback
import asyncio
import json
import time
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
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

# --- Pre-create LLM clients at startup for reuse ---
_clients = {}

def _get_client(model_name: str):
    """Lazily create and cache LLM clients."""
    if model_name not in _clients:
        if model_name == "gpt-5.2-chat":
            _clients[model_name] = AzureOpenAI(
                api_version=os.getenv("GPT52_API_VERSION", "2025-04-01-preview"),
                azure_endpoint=os.getenv("GPT52_ENDPOINT"),
                api_key=os.getenv("GPT52_KEY"),
            )
        elif model_name == "phi-4-mini-reasoning":
            _clients[model_name] = OpenAI(
                base_url=os.getenv("PHI4_ENDPOINT"),
                api_key=os.getenv("PHI4_KEY"),
            )
        elif model_name == "deepseek-v3.2":
            _clients[model_name] = OpenAI(
                base_url=os.getenv("DEEPSEEK_ENDPOINT"),
                api_key=os.getenv("DEEPSEEK_KEY"),
            )
    return _clients[model_name]

MODEL_CONFIG = {
    "gpt-5.2-chat": {"deployment": "GPT52_DEPLOYMENT", "default": "gpt-5.2-chat", "max_tokens_key": "max_completion_tokens", "max_tokens": 1024},
    "phi-4-mini-reasoning": {"deployment": "PHI4_DEPLOYMENT", "default": "Phi-4-mini-reasoning", "max_tokens_key": "max_tokens", "max_tokens": 512},
    "deepseek-v3.2": {"deployment": "DEEPSEEK_DEPLOYMENT", "default": "DeepSeek-V3.2", "max_tokens_key": "max_tokens", "max_tokens": 1024},
}

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
    config = MODEL_CONFIG.get(model_name.lower())
    if not config:
        return f"Model {model_name} not supported."
    
    deployment = os.getenv(config["deployment"], config["default"])
    try:
        client = _get_client(model_name.lower())
        kwargs = {
            "messages": [{"role": "user", "content": prompt}],
            "model": deployment,
            config["max_tokens_key"]: config["max_tokens"],
        }
        response = client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content
        if content is None:
            return f"{model_name} returned empty response (None)"
        stripped = content.strip()
        return stripped if stripped else f"{model_name} returned empty content"
    except Exception as e:
        logger.error(f"{model_name} API error: {type(e).__name__}: {str(e)}")
        return f"{model_name} API error: {type(e).__name__}: {str(e)}"

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
    
    # Call all 3 models in parallel
    logger.info("Calling all models in parallel")
    results = await asyncio.gather(
        asyncio.to_thread(query_llm, models[0], prompt),
        asyncio.to_thread(query_llm, models[1], prompt),
        asyncio.to_thread(query_llm, models[2], prompt),
    )
    responses = dict(zip(models, results))
    for model in models:
        logger.info(f"Model {model} response length: {len(responses[model])}")
    
    logger.info("All responses collected")
    best_model, grades, reasons, judge_summary = await asyncio.to_thread(triage_responses, prompt, responses)
    
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

# --- SSE streaming endpoint: sends results as each model finishes ---
@app.post("/api/llm/stream")
async def api_llm_stream(request: AskRequest):
    prompt = request.query or request.prompt
    if not prompt:
        return {"error": "Please provide either 'prompt' or 'query' field"}

    async def event_stream():
        start = time.time()
        models = ["gpt-5.2-chat", "phi-4-mini-reasoning", "deepseek-v3.2"]
        responses = {}

        # Create async tasks for all models
        async def call_model(model):
            t0 = time.time()
            result = await asyncio.to_thread(query_llm, model, prompt)
            elapsed = round(time.time() - t0, 1)
            return model, result, elapsed

        tasks = [asyncio.create_task(call_model(m)) for m in models]

        # Yield each model's result as it finishes
        for coro in asyncio.as_completed(tasks):
            model, answer, elapsed = await coro
            responses[model] = answer
            event_data = json.dumps({"model": model, "answer": answer, "elapsed": elapsed})
            yield f"event: model_result\ndata: {event_data}\n\n"

        # Now run the judge
        judge_start = time.time()
        best_model, grades, reasons, judge_summary = await asyncio.to_thread(triage_responses, prompt, responses)
        judge_elapsed = round(time.time() - judge_start, 1)
        total_elapsed = round(time.time() - start, 1)

        judge_data = json.dumps({
            "best": best_model,
            "grades": grades,
            "reasons": reasons,
            "judge_summary": judge_summary,
            "judge": "gpt-5.2-chat",
            "judge_elapsed": judge_elapsed,
            "total_elapsed": total_elapsed,
        })
        yield f"event: judge_result\ndata: {judge_data}\n\n"
        yield f"event: done\ndata: {{}}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")

# TODO: Add tracing/debugging instrumentation
# TODO: Integrate with agentdev and VSCode debugging

# Mount static files at root path to serve React assets
app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8087, reload=True)
