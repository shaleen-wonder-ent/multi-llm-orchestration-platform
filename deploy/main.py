
from dotenv import load_dotenv
import os
load_dotenv()
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import openai
import requests

app = FastAPI()

# Allow CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (React build)
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.isdir(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Serve index.html for frontend
@app.get("/")
async def serve_index():
    index_path = os.path.join(static_dir, "index.html")
    return FileResponse(index_path)



# LLM orchestration endpoint: call both GPT-5.2 and PHI-4, return both answers, grades, and best
@app.post("/api/llm")
async def llm_endpoint(request: Request):
    data = await request.json()
    prompt = data.get("prompt", "")

    def call_azure_llm(endpoint, key, deployment, api_version, prompt):
        url = f"{endpoint}/openai/deployments/{deployment}/chat/completions?api-version={api_version}"
        headers = {
            "Content-Type": "application/json",
            "api-key": key
        }
        payload = {
            "messages": [{"role": "user", "content": prompt}],
            "max_completion_tokens": 1024,
            "frequency_penalty": 0,
            "presence_penalty": 0,
        }
        resp = requests.post(url, headers=headers, json=payload)
        resp.raise_for_status()
        result = resp.json()
        return result["choices"][0]["message"]["content"]

    try:
        gpt_answer = call_azure_llm(
            os.getenv("GPT52_ENDPOINT"),
            os.getenv("GPT52_KEY"),
            os.getenv("GPT52_DEPLOYMENT"),
            os.getenv("GPT52_API_VERSION"),
            prompt
        )
    except Exception as e:
        gpt_answer = f"Error: {str(e)}"

    try:
        phi_answer = call_azure_llm(
            os.getenv("PHI4_ENDPOINT"),
            os.getenv("PHI4_KEY"),
            os.getenv("PHI4_DEPLOYMENT"),
            os.getenv("PHI4_API_VERSION"),
            prompt
        )
    except Exception as e:
        phi_answer = f"Error: {str(e)}"

    # Use Deepseek LLM for evaluation
    def call_deepseek_eval(api_url, api_key, prompt, gpt_answer, phi_answer):
        eval_prompt = f"""
You are an expert judge. Given a user prompt and two LLM answers, rate each answer from 1-5 (higher is better) and select which is best. Respond in JSON as: {{"gpt52": <score>, "phi4": <score>, "best": <model>}}

Prompt: {prompt}

GPT-5.2 Answer: {gpt_answer}

PHI-4 Answer: {phi_answer}
"""
        payload = {
            "model": os.getenv("DEEPSEEK_MODEL", "DeepSeek-V3.2"),
            "messages": [
                {"role": "system", "content": "You are a helpful assistant and expert judge."},
                {"role": "user", "content": eval_prompt}
            ]
        }
        headers = {
            "Content-Type": "application/json",
            "api-key": api_key
        }
        resp = requests.post(api_url, headers=headers, json=payload)
        resp.raise_for_status()
        result = resp.json()
        # Try to extract JSON from the response
        import re, json as pyjson
        content = result["choices"][0]["message"]["content"]
        print("DEEPSEEK RAW RESPONSE:\n", content)
        match = re.search(r'\{.*\}', content, re.DOTALL)
        if match:
            return pyjson.loads(match.group(0)), content
        return {"gpt52": 0, "phi4": 0, "best": "unknown"}, content

    deepseek_raw = None
    try:
        deepseek_api_url = os.getenv("DEEPSEEK_API_URL", "https://shaleenthapa-6749-resource.services.ai.azure.com/openai/v1/chat/completions")
        deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
        eval_result, deepseek_raw = call_deepseek_eval(
            deepseek_api_url,
            deepseek_api_key,
            prompt,
            gpt_answer,
            phi_answer
        )
        gpt_grade = eval_result.get("gpt52", 0)
        phi_grade = eval_result.get("phi4", 0)
        best = eval_result.get("best", "gpt52")
    except Exception as e:
        gpt_grade = 0
        phi_grade = 0
        best = "gpt52"
        deepseek_raw = str(e)

    return JSONResponse({
        "answers": {
            "gpt52": gpt_answer,
            "phi4": phi_answer
        },
        "grades": {
            "gpt52": gpt_grade,
            "phi4": phi_grade
        },
        "best": best,
        "deepseek_raw": deepseek_raw
    })
