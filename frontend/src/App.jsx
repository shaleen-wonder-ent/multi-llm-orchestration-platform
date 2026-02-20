import React, { useState } from "react";

const ALL_MODELS = ["gpt-5.2-chat", "phi-4-mini-reasoning", "deepseek-v3.2"];

function App() {
  const [prompt, setPrompt] = useState("");
  const [answers, setAnswers] = useState({});
  const [judgeData, setJudgeData] = useState(null);
  const [judgeModel, setJudgeModel] = useState(null);
  const [contestants, setContestants] = useState([]);
  const [loading, setLoading] = useState(false);
  const [pendingModels, setPendingModels] = useState([]);
  const [error, setError] = useState("");

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError("");
    setAnswers({});
    setJudgeData(null);
    setJudgeModel(null);
    setContestants([]);
    setPendingModels([...ALL_MODELS]);

    try {
      const res = await fetch("/api/llm/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt }),
      });
      if (!res.ok) throw new Error("Server error");

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        const lines = buffer.split("\n");
        buffer = "";
        let eventType = "";
        for (const line of lines) {
          if (line.startsWith("event: ")) {
            eventType = line.slice(7).trim();
          } else if (line.startsWith("data: ")) {
            const data = JSON.parse(line.slice(6));
            if (eventType === "judge_selected") {
              setJudgeModel(data.judge);
              setContestants(data.contestants);
              setPendingModels(data.contestants);
            } else if (eventType === "model_result") {
              setAnswers(prev => ({ ...prev, [data.model]: { answer: data.answer, elapsed: data.elapsed } }));
              setPendingModels(prev => prev.filter(m => m !== data.model));
            } else if (eventType === "judge_result") {
              setJudgeData(data);
            }
            eventType = "";
          } else if (line.trim() === "") {
            // empty line between events
          } else {
            buffer += line + "\n";
          }
        }
      }
    } catch (err) {
      setError("Failed to get response. " + err.message);
    }
    setLoading(false);
  };

  const hasAnswers = Object.keys(answers).length > 0;
  const displayModels = contestants.length > 0 ? contestants : ALL_MODELS;

  return (
    <div style={{ maxWidth: 1100, margin: "40px auto", padding: 24, borderRadius: 8, boxShadow: "0 2px 8px #ddd", color: "#213547", background: "#fff" }}>
      <h2>Multi-LLM Orchestration App</h2>
      <form onSubmit={handleSubmit} style={{ marginBottom: 24 }}>
        <input
          id="prompt-input"
          name="prompt"
          type="text"
          value={prompt}
          onChange={e => setPrompt(e.target.value)}
          placeholder="Enter your prompt..."
          style={{ width: "100%", padding: 8, fontSize: 16, marginBottom: 8 }}
        />
        <button type="submit" disabled={loading} style={{ padding: "8px 16px", fontSize: 16 }}>
          {loading ? "Loading..." : "Ask"}
        </button>
      </form>
      {error && <div style={{ color: "red", marginBottom: 16 }}>{error}</div>}
      {(hasAnswers || loading) && (
        <div>
          {judgeModel && (
            <div style={{ fontSize: 13, color: "#722ed1", marginBottom: 12, background: "#f9f0ff", border: "1px solid #d3adf7", borderRadius: 4, padding: "8px 12px" }}>
              <strong>Judge this round:</strong> {judgeModel.toUpperCase()} &nbsp;|&nbsp; <strong>Contestants:</strong> {displayModels.map(m => m.toUpperCase()).join(" vs ")}
            </div>
          )}
          <h3>Contestant Answers</h3>
          <div style={{ display: "flex", gap: 16, marginBottom: 16 }}>
            {displayModels.map(model => {
              const data = answers[model];
              const isPending = pendingModels.includes(model);
              const isBest = judgeData && judgeData.best === model;
              return (
                <div key={model} style={{
                  flex: 1,
                  background: isBest ? "#fffbe6" : isPending ? "#f9f9f9" : "#e6f7ff",
                  color: "#213547",
                  padding: 12,
                  borderRadius: 4,
                  border: isBest ? "2px solid #faad14" : "1px solid #d9d9d9",
                  whiteSpace: "pre-wrap",
                  opacity: isPending ? 0.6 : 1,
                  transition: "all 0.3s ease",
                }}>
                  <strong style={{ fontSize: 16 }}>{model.toUpperCase()}</strong>
                  {isPending && (
                    <div style={{ fontSize: 13, color: "#999", margin: "8px 0" }}>Waiting for response...</div>
                  )}
                  {data && (
                    <>
                      <div style={{ fontSize: 11, color: "#888", margin: "2px 0 4px 0" }}>
                        Responded in {data.elapsed}s
                      </div>
                      {judgeData && judgeData.grades[model] !== undefined && (
                        <div style={{ fontSize: 13, margin: "4px 0 8px 0" }}>
                          Grade: <b>{judgeData.grades[model]}/10</b> {isBest && <span style={{ color: "#faad14" }}>(Best)</span>}
                        </div>
                      )}
                      {judgeData && judgeData.reasons && judgeData.reasons[model] && (
                        <div style={{ fontSize: 12, color: "#555", background: "#f5f5f5", padding: "6px 8px", borderRadius: 4, marginBottom: 8, fontStyle: "italic" }}>
                          Judge: {judgeData.reasons[model]}
                        </div>
                      )}
                      {formatLLMAnswer(data.answer)}
                    </>
                  )}
                </div>
              );
            })}
          </div>
          {loading && !judgeData && Object.keys(answers).length === displayModels.length && (
            <div style={{ fontSize: 13, color: "#1890ff", marginBottom: 12 }}>
              {judgeModel ? `${judgeModel.toUpperCase()} is evaluating responses...` : "Evaluating responses..."}
            </div>
          )}
          {judgeData && judgeData.judge_summary && (
            <div style={{ background: "#f0f9ff", border: "1px solid #91d5ff", borderRadius: 4, padding: "10px 14px", marginBottom: 12, color: "#213547" }}>
              <strong style={{ fontSize: 13 }}>Judge ({judgeData.judge}):</strong>
              <span style={{ fontSize: 13, marginLeft: 6 }}>{judgeData.judge_summary}</span>
              <span style={{ fontSize: 11, color: "#888", marginLeft: 8 }}>({judgeData.judge_elapsed}s)</span>
            </div>
          )}
          {judgeData && (
            <div style={{ fontWeight: 500, color: "#52c41a" }}>
              Best LLM: {judgeData.best.toUpperCase()}
              <span style={{ fontSize: 12, color: "#888", fontWeight: 400, marginLeft: 8 }}>Total: {judgeData.total_elapsed}s</span>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// Helper to format LLM output: remove <think>...</think>, \boxed{...}, and render **bold**
function formatLLMAnswer(text) {
  if (!text) return null;
  // Remove <think>...</think> blocks
  text = text.replace(/<think>[\s\S]*?<\/think>/gi, "");
  // Replace \boxed{...} with just ...
  text = text.replace(/\\boxed\{([^}]*)\}/g, "$1");
  // Replace **bold** with <strong>bold</strong>
  text = text.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
  // Replace multiple newlines with <br/>
  text = text.replace(/\n{2,}/g, '<br/><br/>');
  text = text.replace(/\n/g, '<br/>');
  // Render as HTML (dangerouslySetInnerHTML)
  return <span dangerouslySetInnerHTML={{ __html: text.trim() }} />;
}

export default App;
