
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

    # Mock grading: you can replace this with your own logic
    gpt_grade = 4
    phi_grade = 5
    best = "phi4" if phi_grade >= gpt_grade else "gpt52"

    return JSONResponse({
        "answers": {
            "gpt52": gpt_answer,
            "phi4": phi_answer
        },
        "grades": {
            "gpt52": gpt_grade,
            "phi4": phi_grade
        },
        "best": best
    })
