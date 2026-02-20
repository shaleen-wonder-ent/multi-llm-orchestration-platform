import React, { useState } from "react";

function App() {
  const [prompt, setPrompt] = useState("");
  const [responses, setResponses] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError("");
    setResponses(null);
    try {
      const res = await fetch("/api/llm", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt }),
      });
      if (!res.ok) throw new Error("Server error");
      const data = await res.json();
      setResponses(data);
    } catch (err) {
      setError("Failed to get response. " + err.message);
    }
    setLoading(false);
  };

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
      {responses && (
        <div>
          <h3>LLM Answers</h3>
          <div style={{ display: "flex", gap: 16, marginBottom: 16 }}>
            {Object.entries(responses.answers).map(([model, ans]) => (
              <div key={model} style={{
                flex: 1,
                background: responses.best === model ? "#fffbe6" : "#e6f7ff",
                color: "#213547",
                padding: 12,
                borderRadius: 4,
                border: responses.best === model ? "2px solid #faad14" : "1px solid #91d5ff",
                whiteSpace: "pre-wrap"
              }}>
                <strong style={{ fontSize: 16 }}>{model.toUpperCase()}</strong>
                <div style={{ fontSize: 13, margin: "4px 0 8px 0" }}>
                  Grade: <b>{responses.grades[model]}/10</b> {responses.best === model && <span style={{ color: "#faad14" }}>(Best)</span>}
                </div>
                {responses.reasons && responses.reasons[model] && (
                  <div style={{ fontSize: 12, color: "#555", background: "#f5f5f5", padding: "6px 8px", borderRadius: 4, marginBottom: 8, fontStyle: "italic" }}>
                    Judge: {responses.reasons[model]}
                  </div>
                )}
                {formatLLMAnswer(ans)}
              </div>
            ))}
          </div>
          {responses.judge_summary && (
            <div style={{ background: "#f0f9ff", border: "1px solid #91d5ff", borderRadius: 4, padding: "10px 14px", marginBottom: 12, color: "#213547" }}>
              <strong style={{ fontSize: 13 }}>Judge ({responses.judge || "GPT-5.2"}):</strong>
              <span style={{ fontSize: 13, marginLeft: 6 }}>{responses.judge_summary}</span>
            </div>
          )}
          <div style={{ fontWeight: 500, color: "#52c41a" }}>
            Best LLM: {responses.best.toUpperCase()}
          </div>
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
