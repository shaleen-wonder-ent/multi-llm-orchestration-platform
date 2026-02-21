"""
Multi-LLM Orchestration App — LangChain Branch
================================================
Flow:
  Round 0  : All 3 contestant LLMs answer. Judge scores them. User picks best model.
  Rounds 1-3: Chosen model answer shown. All 3 still run in background.
              Judge scores silently. If another model beats chosen, user gets a
              non-intrusive switch-hint banner.
  After R3  : Highest cumulative-score model locks in as primary for the session.
"""

import os
import logging
import asyncio
import json
import time
import random
import re
import uuid
from typing import Optional, Dict

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from openai import AzureOpenAI, OpenAI
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# 
#  LLM client & model config
# 
_clients: Dict[str, object] = {}

def _get_client(model_name: str):
    if model_name not in _clients:
        if model_name == "gpt-5.2-chat":
            _clients[model_name] = AzureOpenAI(
                api_version=os.getenv("GPT52_API_VERSION", "2024-12-01-preview"),
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
    return _clients.get(model_name)

MODEL_CONFIG = {
    "gpt-5.2-chat": {
        "deployment": "GPT52_DEPLOYMENT",
        "default": "gpt-5.2-chat",
        "max_tokens_key": "max_completion_tokens",
        "max_tokens": 1024,
    },
    "phi-4-mini-reasoning": {
        "deployment": "PHI4_DEPLOYMENT",
        "default": "phi-4-mini-reasoning",
        "max_tokens_key": "max_tokens",
        "max_tokens": 512,
    },
    "deepseek-v3.2": {
        "deployment": "DEEPSEEK_DEPLOYMENT",
        "default": "DeepSeek-V3.2",
        "max_tokens_key": "max_tokens",
        "max_tokens": 1024,
    },
}

ALL_MODELS = list(MODEL_CONFIG.keys())
MAX_RECURSIONS = 3  # background comparison rounds before lock-in


# 
#  In-memory session store
# 
_sessions: Dict[str, dict] = {}


def _new_session(session_id: str) -> dict:
    sess = {
        "chosen_model": None,
        "recursion_count": 0,
        "cumulative_scores": {m: 0.0 for m in ALL_MODELS},
        "locked": False,
    }
    _sessions[session_id] = sess
    return sess


def _get_or_create_session(session_id: Optional[str]):
    if not session_id or session_id not in _sessions:
        session_id = session_id or str(uuid.uuid4())
        return session_id, _new_session(session_id)
    return session_id, _sessions[session_id]


# 
#  Core LLM call
# 
def query_llm(model_name: str, prompt: str, history: list | None = None) -> str:
    config = MODEL_CONFIG.get(model_name.lower())
    if not config:
        return f"[Model {model_name} not supported]"
    deployment = os.getenv(config["deployment"], config["default"])
    messages = list(history) if history else []
    messages.append({"role": "user", "content": prompt})
    try:
        client = _get_client(model_name.lower())
        if client is None:
            return f"[Client for {model_name} could not be created — check .env]"
        kwargs = {
            "messages": messages,
            "model": deployment,
            config["max_tokens_key"]: config["max_tokens"],
        }
        response = client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content
        if not content:
            return f"[{model_name} returned empty response]"
        content = re.sub(r"<think>[\s\S]*?</think>", "", content).strip()
        return content or f"[{model_name} returned empty after stripping reasoning]"
    except Exception as e:
        logger.error(f"{model_name} error: {e}")
        return f"[{model_name} error: {type(e).__name__}: {str(e)}]"


# 
#  Judge
# 
def judge_responses(prompt: str, responses: dict, judge_model: str) -> tuple:
    """Returns (best_model, grades_dict, reasons_dict, summary_str)"""
    responses_text = ""
    for model, resp in responses.items():
        truncated = resp[:1500] if len(resp) > 1500 else resp
        responses_text += f"\n--- {model} ---\n{truncated}\n"

    judge_prompt = f"""You are an expert judge evaluating AI responses.
User question: "{prompt}"

Responses:
{responses_text}

Score each model 1-10 on accuracy, completeness, clarity, and relevance.
Return ONLY valid JSON (no markdown, no code fences):
{{
  "evaluations": {{
    "<model_name>": {{"grade": <1-10>, "reason": "<1-2 sentences>"}}
  }},
  "best": "<model_name>",
  "summary": "<1 sentence>"
}}"""

    config = MODEL_CONFIG[judge_model]
    deployment = os.getenv(config["deployment"], config["default"])
    judge_max = max(config["max_tokens"], 1024)
    try:
        client = _get_client(judge_model)
        kwargs = {
            "messages": [{"role": "user", "content": judge_prompt}],
            "model": deployment,
            config["max_tokens_key"]: judge_max,
        }
        raw = client.chat.completions.create(**kwargs).choices[0].message.content or ""
        raw = re.sub(r"<think>[\s\S]*?</think>", "", raw).strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
            raw = raw.rsplit("```", 1)[0].strip()
        match = re.search(r"\{[\s\S]*\}", raw)
        if match:
            raw = match.group(0)
        data = json.loads(raw)
        evals = data.get("evaluations", {})
        grades = {m: float(evals[m]["grade"]) for m in evals if m in responses}
        reasons = {m: evals[m].get("reason", "") for m in evals if m in responses}
        best = data.get("best", max(grades, key=lambda k: grades[k]))
        summary = data.get("summary", "")
        return best, grades, reasons, summary
    except Exception as e:
        logger.warning(f"Judge ({judge_model}) failed: {e} — using heuristic fallback")

    # Heuristic fallback
    grades, reasons = {}, {}
    for m, resp in responses.items():
        if resp.startswith("[") and "error" in resp.lower():
            grades[m] = 1.0
            reasons[m] = "Response contained an error."
        else:
            score = min(len(resp) / 120.0, 5.0)
            bonus = 3.0 if any(w in resp.lower() for w in ["because", "therefore", "however", "specifically"]) else 1.0
            grades[m] = min(score + bonus, 10.0)
            reasons[m] = "Scored by heuristic (judge LLM unavailable)."
    best = max(grades, key=lambda k: grades[k])
    return best, grades, reasons, "Scored by heuristic fallback."


# 
#  FastAPI app
# 
app = FastAPI(title="Multi-LLM Orchestration — LangChain Branch")


@app.get("/")
def serve_index():
    return FileResponse("static/index.html")


@app.get("/status")
def status():
    return {"status": "running", "branch": "langchain", "models": ALL_MODELS}


@app.get("/health")
def health():
    return {"status": "healthy", "branch": "langchain"}


# 
#  Request models
# 
class ChatRequest(BaseModel):
    prompt: str
    session_id: Optional[str] = None
    chosen_model: Optional[str] = None
    history: Optional[list] = None


class SelectModelRequest(BaseModel):
    session_id: str
    model: str


# 
#  /api/select-model
# 
@app.post("/api/select-model")
async def select_model(req: SelectModelRequest):
    _, sess = _get_or_create_session(req.session_id)
    if req.model not in ALL_MODELS:
        return {"error": f"Unknown model: {req.model}"}
    sess["chosen_model"] = req.model
    sess["recursion_count"] = 0
    sess["locked"] = False
    return {"session_id": req.session_id, "chosen_model": req.model}


# 
#  /api/chat/stream  — main SSE endpoint
# 
@app.post("/api/chat/stream")
async def chat_stream(req: ChatRequest):
    """
    SSE events:
      session         — initial session state
      judge_selected  — which model is judging
      model_result    — (tournament mode) each model answer as it arrives
      judge_result    — scores, best, summary
      background_result — (chat mode) all model answers with grades
      shown_answer    — (chat mode) the chosen model answer to display
      switch_hint     — (chat mode) if another model scored significantly better
      locked_in       — cumulative winner after MAX_RECURSIONS rounds
      done            — stream complete
    """
    prompt = req.prompt.strip()
    if not prompt:
        async def _err():
            yield f"event: error\ndata: {json.dumps({'message': 'Empty prompt'})}\n\n"
        return StreamingResponse(_err(), media_type="text/event-stream")

    session_id, sess = _get_or_create_session(req.session_id)

    if req.chosen_model and req.chosen_model in ALL_MODELS:
        sess["chosen_model"] = req.chosen_model

    chosen = sess["chosen_model"]
    locked = sess["locked"]
    recursion_count = sess["recursion_count"]
    mode = "tournament" if chosen is None else "chat"
    history = req.history or []

    async def event_stream():
        nonlocal chosen, locked, recursion_count

        yield f"event: session\ndata: {json.dumps({'session_id': session_id, 'mode': mode, 'chosen_model': chosen, 'recursion_count': recursion_count, 'locked': locked, 'max_recursions': MAX_RECURSIONS})}\n\n"

        if mode == "tournament":
            judge_model = random.choice(ALL_MODELS)
            contestants = ALL_MODELS
        else:
            non_chosen = [m for m in ALL_MODELS if m != chosen]
            judge_model = random.choice(non_chosen)
            contestants = ALL_MODELS

        yield f"event: judge_selected\ndata: {json.dumps({'judge': judge_model, 'contestants': contestants, 'mode': mode})}\n\n"

        responses: Dict[str, str] = {}
        elapsed_map: Dict[str, float] = {}

        async def call_model(model: str):
            t0 = time.time()
            try:
                answer = await asyncio.wait_for(
                    asyncio.to_thread(query_llm, model, prompt, history),
                    timeout=45.0
                )
            except asyncio.TimeoutError:
                answer = f"[{model} timed out after 45s]"
            return model, answer, round(time.time() - t0, 1)

        if mode == "tournament":
            tasks = [asyncio.create_task(call_model(m)) for m in contestants]
            for coro in asyncio.as_completed(tasks):
                model, answer, elapsed = await coro
                responses[model] = answer
                elapsed_map[model] = elapsed
                yield f"event: model_result\ndata: {json.dumps({'model': model, 'answer': answer, 'elapsed': elapsed})}\n\n"
        else:
            # Chat mode: yield chosen model's answer immediately, then run background models
            c_model, c_answer, c_elapsed = await call_model(chosen)
            responses[c_model] = c_answer
            elapsed_map[c_model] = c_elapsed
            yield f"event: shown_answer\ndata: {json.dumps({'model': chosen, 'answer': c_answer, 'elapsed': c_elapsed})}\n\n"
            bg_tasks = [asyncio.create_task(call_model(m)) for m in ALL_MODELS if m != chosen]
            for coro in asyncio.as_completed(bg_tasks):
                model, answer, elapsed = await coro
                responses[model] = answer
                elapsed_map[model] = elapsed

        judge_start = time.time()
        best_model, grades, reasons, summary = await asyncio.to_thread(
            judge_responses, prompt, responses, judge_model
        )
        judge_elapsed = round(time.time() - judge_start, 1)

        for m, g in grades.items():
            sess["cumulative_scores"][m] = sess["cumulative_scores"].get(m, 0.0) + g

        yield f"event: judge_result\ndata: {json.dumps({'best': best_model, 'grades': grades, 'reasons': reasons, 'judge_summary': summary, 'judge': judge_model, 'judge_elapsed': judge_elapsed, 'cumulative_scores': sess['cumulative_scores']})}\n\n"

        if mode == "chat":
            for model, answer in responses.items():
                yield f"event: background_result\ndata: {json.dumps({'model': model, 'answer': answer, 'elapsed': elapsed_map[model], 'grade': grades.get(model)})}\n\n"

            # shown_answer was already emitted before judge ran
            chosen_grade = grades.get(chosen, 0)
            better_candidates = {
                m: g for m, g in grades.items()
                if m != chosen and g > chosen_grade + 1.0
            }
            if better_candidates:
                better_model = max(better_candidates, key=lambda k: better_candidates[k])
                yield f"event: switch_hint\ndata: {json.dumps({'better_model': better_model, 'better_grade': better_candidates[better_model], 'current_model': chosen, 'current_grade': chosen_grade, 'reason': reasons.get(better_model, '')})}\n\n"

            sess["recursion_count"] += 1
            recursion_count = sess["recursion_count"]

            if recursion_count >= MAX_RECURSIONS and not locked:
                cumulative = sess["cumulative_scores"]
                locked_model = max(cumulative, key=lambda k: cumulative[k])
                sess["locked"] = True
                sess["chosen_model"] = locked_model
                locked = True
                chosen = locked_model
                yield f"event: locked_in\ndata: {json.dumps({'model': locked_model, 'cumulative_scores': cumulative, 'message': f'{locked_model} locked in as primary model (best across {MAX_RECURSIONS} rounds).'})}\n\n"

        yield f"event: done\ndata: {{}}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# 
#  Legacy endpoints for compatibility
# 
class AskRequest(BaseModel):
    prompt: str = None
    query: str = None


@app.post("/ask")
@app.post("/api/llm")
async def legacy_ask(request: AskRequest):
    p = request.query or request.prompt
    if not p:
        return {"error": "Provide 'prompt' or 'query'"}
    judge_model = random.choice(ALL_MODELS)
    results = await asyncio.gather(*[asyncio.to_thread(query_llm, m, p) for m in ALL_MODELS])
    responses = dict(zip(ALL_MODELS, results))
    best, grades, reasons, summary = await asyncio.to_thread(judge_responses, p, responses, judge_model)
    return {"prompt": p, "answers": responses, "best": best, "grades": grades,
            "reasons": reasons, "judge_summary": summary, "judge": judge_model,
            "best_answer": responses[best]}


# Mount static files last
app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8087, reload=True)
