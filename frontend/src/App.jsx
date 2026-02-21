import React, { useState, useRef, useEffect } from "react";

const ALL_MODELS = ["gpt-5.2-chat", "phi-4-mini-reasoning", "deepseek-v3.2"];
const MAX_RECURSIONS = 3;

//  Utility 
function formatAnswer(text) {
  if (!text) return null;
  text = text.replace(/<think>[\s\S]*?<\/think>/gi, "");
  text = text.replace(/\\boxed\{([^}]*)\}/g, "$1");
  text = text.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  text = text.replace(/\n{2,}/g, "<br/><br/>");
  text = text.replace(/\n/g, "<br/>");
  return <span dangerouslySetInnerHTML={{ __html: text.trim() }} />;
}

const modelColor = {
  "gpt-5.2-chat":        { bg: "#e6f4ff", border: "#91caff", badge: "#1677ff" },
  "phi-4-mini-reasoning":{ bg: "#f6ffed", border: "#95de64", badge: "#52c41a" },
  "deepseek-v3.2":       { bg: "#fff7e6", border: "#ffd591", badge: "#fa8c16" },
};

function ModelBadge({ model, isBest, grade }) {
  const c = modelColor[model] || { badge: "#888" };
  return (
    <span style={{
      display: "inline-flex", alignItems: "center", gap: 4,
      background: c.badge, color: "#fff", borderRadius: 12,
      padding: "2px 10px", fontSize: 12, fontWeight: 600,
    }}>
      {model.toUpperCase()}
      {grade !== undefined && <span style={{ opacity: 0.85 }}> {grade}/10</span>}
      {isBest && <span style={{ marginLeft: 2 }}></span>}
    </span>
  );
}

//  Tournament View 
function TournamentView({ tournamentData, onSelectModel }) {
  const { answers, judgeData, judgeModel, loading, pendingModels } = tournamentData;

  return (
    <div>
      <div style={{ fontSize: 13, color: "#722ed1", marginBottom: 12, background: "#f9f0ff",
        border: "1px solid #d3adf7", borderRadius: 6, padding: "8px 14px" }}>
         <strong>Tournament Round</strong> — Judge: <strong>{judgeModel?.toUpperCase()}</strong>
        &nbsp;|&nbsp; All 3 models compete. Pick your preferred answer to continue.
      </div>

      <div style={{ display: "flex", gap: 14, marginBottom: 16 }}>
        {ALL_MODELS.map(model => {
          const data = answers[model];
          const isPending = pendingModels.includes(model);
          const isBest = judgeData?.best === model;
          const grade = judgeData?.grades?.[model];
          const c = modelColor[model] || { bg: "#f9f9f9", border: "#ddd" };
          return (
            <div key={model} style={{
              flex: 1, background: isBest ? "#fffbe6" : c.bg,
              border: `2px solid ${isBest ? "#faad14" : c.border}`,
              borderRadius: 8, padding: 14,
              opacity: isPending ? 0.55 : 1,
              transition: "all 0.3s ease",
            }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
                <ModelBadge model={model} isBest={isBest} grade={grade !== undefined ? Math.round(grade) : undefined} />
                {data && <span style={{ fontSize: 11, color: "#888" }}>{data.elapsed}s</span>}
              </div>

              {isPending && <div style={{ color: "#aaa", fontSize: 13 }}> Waiting</div>}

              {judgeData?.reasons?.[model] && (
                <div style={{ fontSize: 12, color: "#555", background: "rgba(0,0,0,0.04)",
                  padding: "5px 8px", borderRadius: 4, marginBottom: 8, fontStyle: "italic" }}>
                  Judge: {judgeData.reasons[model]}
                </div>
              )}

              {data && (
                <div style={{ fontSize: 14, color: "#213547", whiteSpace: "pre-wrap",
                  lineHeight: 1.6, maxHeight: 260, overflowY: "auto" }}>
                  {formatAnswer(data.answer)}
                </div>
              )}

              {judgeData && data && (
                <button onClick={() => onSelectModel(model)} style={{
                  marginTop: 12, width: "100%", padding: "7px 0",
                  background: isBest ? "#faad14" : "#1677ff", color: "#fff",
                  border: "none", borderRadius: 6, cursor: "pointer", fontWeight: 600, fontSize: 13,
                }}>
                  {isBest ? " Use this (Best)" : "Use this model"}
                </button>
              )}
            </div>
          );
        })}
      </div>

      {loading && !judgeData && Object.keys(answers).length < ALL_MODELS.length && (
        <div style={{ color: "#888", fontSize: 13, textAlign: "center", padding: 8 }}>
          Collecting responses ({Object.keys(answers).length}/{ALL_MODELS.length})
        </div>
      )}
      {loading && !judgeData && Object.keys(answers).length === ALL_MODELS.length && (
        <div style={{ color: "#1677ff", fontSize: 13, textAlign: "center", padding: 8 }}>
           {judgeModel?.toUpperCase()} is evaluating
        </div>
      )}
      {judgeData?.judge_summary && (
        <div style={{ background: "#f0f9ff", border: "1px solid #91d5ff", borderRadius: 6,
          padding: "10px 14px", fontSize: 13, color: "#213547" }}>
          <strong>Judge verdict:</strong> {judgeData.judge_summary}
        </div>
      )}
    </div>
  );
}

//  Chat Message 
function ChatMessage({ msg }) {
  const [showBackground, setShowBackground] = useState(false);
  if (msg.role === "user") {
    return (
      <div style={{ display: "flex", justifyContent: "flex-end", marginBottom: 12 }}>
        <div style={{ background: "#1677ff", color: "#fff", borderRadius: "18px 18px 4px 18px",
          padding: "10px 16px", maxWidth: "70%", fontSize: 14, lineHeight: 1.6 }}>
          {msg.content}
        </div>
      </div>
    );
  }

  // Assistant message
  const { model, answer, grade, judgeData, backgroundResults, switchHint, lockedIn } = msg;
  const c = modelColor[model] || { bg: "#f5f5f5", border: "#ddd", badge: "#888" };

  return (
    <div style={{ marginBottom: 16 }}>
      {/* Switch hint banner */}
      {switchHint && (
        <div style={{ background: "#fff7e6", border: "1px solid #ffd591", borderRadius: 6,
          padding: "8px 14px", marginBottom: 8, fontSize: 13, display: "flex",
          justifyContent: "space-between", alignItems: "center" }}>
          <span>
             <strong>{switchHint.better_model.toUpperCase()}</strong> scored higher this round
            ({switchHint.better_grade}/10 vs {switchHint.current_grade}/10): {switchHint.reason}
          </span>
          <span style={{ color: "#888", fontSize: 11, marginLeft: 12 }}>
            (You can switch via the selector above)
          </span>
        </div>
      )}

      {/* Lock-in notification */}
      {lockedIn && (
        <div style={{ background: "#f6ffed", border: "1px solid #95de64", borderRadius: 6,
          padding: "8px 14px", marginBottom: 8, fontSize: 13, fontWeight: 600, color: "#237804" }}>
           {lockedIn.message}
        </div>
      )}

      {/* Main answer bubble */}
      <div style={{ display: "flex", alignItems: "flex-start", gap: 10 }}>
        <div style={{ background: c.badge, color: "#fff", borderRadius: 8,
          padding: "4px 8px", fontSize: 11, fontWeight: 700, minWidth: 60, textAlign: "center", marginTop: 4 }}>
          {model?.split("-")[0]?.toUpperCase()}
        </div>
        <div style={{ background: c.bg, border: `1px solid ${c.border}`, borderRadius: "4px 18px 18px 18px",
          padding: "12px 16px", maxWidth: "75%", fontSize: 14, lineHeight: 1.6, color: "#213547" }}>
          {formatAnswer(answer)}
          {grade !== undefined && (
            <div style={{ marginTop: 8, fontSize: 11, color: "#888" }}>
              Judge score this round: <strong>{grade}/10</strong>
            </div>
          )}
        </div>
      </div>

      {/* Background results toggle */}
      {backgroundResults && Object.keys(backgroundResults).length > 0 && (
        <div style={{ marginLeft: 70, marginTop: 8 }}>
          <button onClick={() => setShowBackground(p => !p)} style={{
            background: "none", border: "1px solid #d9d9d9", borderRadius: 4,
            cursor: "pointer", fontSize: 12, color: "#888", padding: "3px 10px",
          }}>
            {showBackground ? "Hide" : "Show"} background model scores
          </button>
          {showBackground && (
            <div style={{ display: "flex", gap: 10, marginTop: 8, flexWrap: "wrap" }}>
              {Object.entries(backgroundResults).map(([m, bdata]) => (
                <div key={m} style={{ border: "1px solid #e8e8e8", borderRadius: 6,
                  padding: "8px 12px", fontSize: 12, background: "#fafafa", minWidth: 180 }}>
                  <ModelBadge model={m} grade={bdata.grade !== undefined ? Math.round(bdata.grade) : undefined} />
                  <div style={{ marginTop: 6, color: "#555", maxHeight: 100, overflowY: "auto", whiteSpace: "pre-wrap" }}>
                    {formatAnswer(bdata.answer)}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

//  Main App 
export default function App() {
  const [sessionId, setSessionId] = useState(null);
  const [chosenModel, setChosenModel] = useState(null);
  const [lockedModel, setLockedModel] = useState(null);
  const [recursionCount, setRecursionCount] = useState(0);
  const [cumScores, setCumScores] = useState({});
  const [mode, setMode] = useState("idle"); // idle | tournament | loading-tournament | chat

  // Tournament state
  const [tournamentPrompt, setTournamentPrompt] = useState("");
  const [tournamentAnswers, setTournamentAnswers] = useState({});
  const [tournamentJudgeData, setTournamentJudgeData] = useState(null);
  const [tournamentJudgeModel, setTournamentJudgeModel] = useState(null);
  const [tournamentPending, setTournamentPending] = useState([]);
  const [tournamentLoading, setTournamentLoading] = useState(false);

  // Chat state
  const [chatHistory, setChatHistory] = useState([]); // [{role,content}] for LLM context
  const [chatMessages, setChatMessages] = useState([]); // UI messages
  const [chatInput, setChatInput] = useState("");
  const [chatLoading, setChatLoading] = useState(false);

  const [error, setError] = useState("");
  const chatEndRef = useRef(null);
  const inputRef = useRef(null);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [chatMessages]);

  //  Submit first question (tournament) 
  const handleTournamentSubmit = async (e) => {
    e.preventDefault();
    if (!tournamentPrompt.trim()) return;
    setError("");
    setTournamentAnswers({});
    setTournamentJudgeData(null);
    setTournamentJudgeModel(null);
    setTournamentPending([...ALL_MODELS]);
    setTournamentLoading(true);
    setMode("loading-tournament");

    try {
      const res = await fetch("/api/chat/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt: tournamentPrompt, session_id: sessionId }),
      });
      if (!res.ok) throw new Error("Server error");

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        // Only process complete SSE events (terminated by \n\n)
        const events = buffer.split("\n\n");
        buffer = events.pop(); // last element is incomplete — keep in buffer
        for (const event of events) {
          let eventType = "";
          let dataLine = "";
          for (const line of event.split("\n")) {
            if (line.startsWith("event: ")) eventType = line.slice(7).trim();
            else if (line.startsWith("data: ")) dataLine = line.slice(6);
          }
          if (!dataLine) continue;
          let data;
          try { data = JSON.parse(dataLine); } catch { continue; }
          if (eventType === "session") {
            setSessionId(data.session_id);
          } else if (eventType === "judge_selected") {
            setTournamentJudgeModel(data.judge);
            setTournamentPending(data.contestants);
            setMode("tournament");
          } else if (eventType === "model_result") {
            setTournamentAnswers(prev => ({ ...prev, [data.model]: { answer: data.answer, elapsed: data.elapsed } }));
            setTournamentPending(prev => prev.filter(m => m !== data.model));
          } else if (eventType === "judge_result") {
            setTournamentJudgeData(data);
            setCumScores(data.cumulative_scores || {});
          }
        }
      }
    } catch (err) {
      setError("Tournament failed: " + err.message);
    }
    setTournamentLoading(false);
  };

  //  User picks a model from tournament 
  const handleSelectModel = async (model) => {
    setError("");
    try {
      await fetch("/api/select-model", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionId, model }),
      });
    } catch (_) { /* ignore, optimistic */ }

    setChosenModel(model);
    setRecursionCount(0);
    // Seed chat history with the tournament Q&A of chosen model
    const firstAnswer = tournamentAnswers[model]?.answer || "";
    setChatHistory([
      { role: "user", content: tournamentPrompt },
      { role: "assistant", content: firstAnswer },
    ]);
    setChatMessages([
      { role: "user", content: tournamentPrompt },
      {
        role: "assistant", model,
        answer: firstAnswer,
        grade: tournamentJudgeData?.grades?.[model],
        judgeData: tournamentJudgeData,
        backgroundResults: null,
        switchHint: null,
        lockedIn: null,
      },
    ]);
    setMode("chat");
  };

  //  Chat follow-up message 
  const handleChatSubmit = async (e) => {
    e.preventDefault();
    if (!chatInput.trim() || chatLoading) return;
    const userMsg = chatInput.trim();
    setChatInput("");
    setError("");
    setChatLoading(true);

    // Add user bubble immediately
    setChatMessages(prev => [...prev, { role: "user", content: userMsg }]);

    // Placeholder for assistant response
    const placeholderIndex = chatMessages.length + 1;
    setChatMessages(prev => [...prev, {
      role: "assistant", model: chosenModel,
      answer: " Thinking", grade: undefined,
      backgroundResults: null, switchHint: null, lockedIn: null, loading: true,
    }]);

    let switchHintData = null;
    let lockedInData = null;
    let shownAnswer = null;
    let shownElapsed = 0;
    let judgeResult = null;
    let backgroundResults = {};

    try {
      const res = await fetch("/api/chat/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prompt: userMsg,
          session_id: sessionId,
          chosen_model: chosenModel,
          history: chatHistory,
        }),
      });
      if (!res.ok) throw new Error("Server error");

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        // Only process complete SSE events (terminated by \n\n)
        const events = buffer.split("\n\n");
        buffer = events.pop(); // last element is incomplete — keep in buffer
        for (const event of events) {
          let eventType = "";
          let dataLine = "";
          for (const line of event.split("\n")) {
            if (line.startsWith("event: ")) eventType = line.slice(7).trim();
            else if (line.startsWith("data: ")) dataLine = line.slice(6);
          }
          if (!dataLine) continue;
          let data;
          try { data = JSON.parse(dataLine); } catch { continue; }
          if (eventType === "judge_result") {
            judgeResult = data;
            setCumScores(data.cumulative_scores || {});
          } else if (eventType === "background_result") {
            backgroundResults[data.model] = { answer: data.answer, elapsed: data.elapsed, grade: data.grade };
          } else if (eventType === "shown_answer") {
            shownAnswer = data.answer;
            shownElapsed = data.elapsed;
          } else if (eventType === "switch_hint") {
            switchHintData = data;
          } else if (eventType === "locked_in") {
            lockedInData = data;
            setLockedModel(data.model);
            setChosenModel(data.model);
          } else if (eventType === "session") {
            setRecursionCount(data.recursion_count);
          }
        }
      }
    } catch (err) {
      setError("Chat error: " + err.message);
    }

    const finalAnswer = shownAnswer || `[No response from ${chosenModel}]`;

    // Update the placeholder bubble
    setChatMessages(prev => {
      const updated = [...prev];
      // find last loading assistant bubble
      const idx = updated.map((m, i) => ({ m, i })).reverse().find(({ m }) => m.role === "assistant" && m.loading)?.i;
      if (idx !== undefined) {
        updated[idx] = {
          role: "assistant",
          model: chosenModel,
          answer: finalAnswer,
          grade: judgeResult?.grades?.[chosenModel],
          judgeData: judgeResult,
          backgroundResults,
          switchHint: switchHintData,
          lockedIn: lockedInData,
          loading: false,
        };
      }
      return updated;
    });

    // Append to LLM history
    setChatHistory(prev => [
      ...prev,
      { role: "user", content: userMsg },
      { role: "assistant", content: finalAnswer },
    ]);

    setChatLoading(false);
    setTimeout(() => inputRef.current?.focus(), 100);
  };

  //  Render 
  return (
    <div style={{ maxWidth: 960, margin: "32px auto", padding: "0 16px", fontFamily: "system-ui, sans-serif", color: "#213547" }}>
      <h2 style={{ textAlign: "center", marginBottom: 4 }}> Multi-LLM Orchestration</h2>
      <p style={{ textAlign: "center", color: "#888", fontSize: 13, marginBottom: 24 }}>
        Ask a question  compare all models  pick your favourite  app tracks the best model for you
      </p>

      {error && (
        <div style={{ background: "#fff2f0", border: "1px solid #ffccc7", borderRadius: 6, padding: "10px 14px", marginBottom: 16, color: "#cf1322", fontSize: 13 }}>
           {error}
        </div>
      )}

      {/*  Status bar (chat mode)  */}
      {mode === "chat" && (
        <div style={{ display: "flex", alignItems: "center", gap: 12, background: "#f9f9f9",
          border: "1px solid #e8e8e8", borderRadius: 8, padding: "10px 16px", marginBottom: 16,
          fontSize: 13, flexWrap: "wrap" }}>
          <span>
            {lockedModel
              ? <strong> Locked: {lockedModel.toUpperCase()}</strong>
              : <>Active model: <ModelBadge model={chosenModel} /></>}
          </span>
          {!lockedModel && (
            <span style={{ color: "#888" }}>
              Background rounds: {recursionCount}/{MAX_RECURSIONS}
            </span>
          )}
          {Object.keys(cumScores).length > 0 && (
            <span style={{ marginLeft: "auto", display: "flex", gap: 8 }}>
              {Object.entries(cumScores).sort((a, b) => b[1] - a[1]).map(([m, s]) => (
                <span key={m} style={{ fontSize: 11, color: "#888" }}>
                  {m.split("-")[0]}: <strong>{Math.round(s)}</strong>
                </span>
              ))}
            </span>
          )}
          {!lockedModel && (
            <select value={chosenModel} onChange={e => setChosenModel(e.target.value)}
              style={{ marginLeft: 8, fontSize: 12, padding: "2px 6px", borderRadius: 4, border: "1px solid #d9d9d9" }}>
              {ALL_MODELS.map(m => <option key={m} value={m}>{m}</option>)}
            </select>
          )}
        </div>
      )}

      {/*  Tournament or idle: first question form  */}
      {(mode === "idle" || mode === "loading-tournament" || mode === "tournament") && (
        <div style={{ marginBottom: 24 }}>
          <form onSubmit={handleTournamentSubmit} style={{ display: "flex", gap: 10, marginBottom: 16 }}>
            <input
              type="text"
              value={tournamentPrompt}
              onChange={e => setTournamentPrompt(e.target.value)}
              placeholder="Ask your first question to start the tournament"
              disabled={tournamentLoading}
              style={{ flex: 1, padding: "10px 14px", fontSize: 15, borderRadius: 8,
                border: "1px solid #d9d9d9", outline: "none" }}
            />
            <button type="submit" disabled={tournamentLoading || !tournamentPrompt.trim()}
              style={{ padding: "10px 22px", background: "#1677ff", color: "#fff",
                border: "none", borderRadius: 8, fontWeight: 600, fontSize: 15, cursor: "pointer" }}>
              {tournamentLoading ? "" : "Ask"}
            </button>
          </form>

          {(mode === "loading-tournament" || mode === "tournament") && (
            <TournamentView
              tournamentData={{
                answers: tournamentAnswers,
                judgeData: tournamentJudgeData,
                judgeModel: tournamentJudgeModel,
                loading: tournamentLoading,
                pendingModels: tournamentPending,
              }}
              onSelectModel={handleSelectModel}
            />
          )}
        </div>
      )}

      {/*  Chat thread  */}
      {mode === "chat" && (
        <div>
          <div style={{ minHeight: 200, maxHeight: 520, overflowY: "auto",
            border: "1px solid #e8e8e8", borderRadius: 10, padding: 16,
            background: "#fafafa", marginBottom: 12 }}>
            {chatMessages.map((msg, i) => <ChatMessage key={i} msg={msg} />)}
            <div ref={chatEndRef} />
          </div>
          <form onSubmit={handleChatSubmit} style={{ display: "flex", gap: 10 }}>
            <input
              ref={inputRef}
              type="text"
              value={chatInput}
              onChange={e => setChatInput(e.target.value)}
              placeholder={`Continue with ${lockedModel || chosenModel}`}
              disabled={chatLoading}
              style={{ flex: 1, padding: "10px 14px", fontSize: 15, borderRadius: 8,
                border: "1px solid #d9d9d9", outline: "none" }}
            />
            <button type="submit" disabled={chatLoading || !chatInput.trim()}
              style={{ padding: "10px 22px", background: "#1677ff", color: "#fff",
                border: "none", borderRadius: 8, fontWeight: 600, fontSize: 15, cursor: "pointer" }}>
              {chatLoading ? "" : "Send"}
            </button>
          </form>
          <div style={{ textAlign: "center", marginTop: 12 }}>
            <button onClick={() => {
              setMode("idle"); setChosenModel(null); setLockedModel(null);
              setSessionId(null); setRecursionCount(0); setCumScores({});
              setChatMessages([]); setChatHistory([]);
              setTournamentPrompt(""); setTournamentAnswers({});
              setTournamentJudgeData(null);
            }} style={{ background: "none", border: "none", color: "#888", cursor: "pointer", fontSize: 12 }}>
               Start a new conversation
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
