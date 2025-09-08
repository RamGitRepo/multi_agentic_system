// src/App.jsx
import React, { useState, useEffect, useRef, useMemo } from "react";
import axios from "axios";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

const API_BASE = import.meta?.env?.VITE_API_BASE || "http://localhost:8000";
const DEBUG = (import.meta?.env?.VITE_DEBUG ?? "false").toString() === "true";

/* ---------- Visual spec ---------- */
const ROLES = {
  user: {
    name: "You",
    emoji: "🧑",
    bubble:
      "bg-gradient-to-br from-indigo-500 via-violet-500 to-fuchsia-500 text-white",
    align: "justify-end",
    corner: "rounded-2xl rounded-br-sm",
  },
  bot: {
    name: "Assistant",
    emoji: "🤖",
    bubble: "bg-white/70 text-slate-900",
    align: "justify-start",
    corner: "rounded-2xl rounded-bl-sm",
  },
};

const uid = () =>
  (typeof crypto !== "undefined" && crypto.randomUUID
    ? crypto.randomUUID()
    : `${Date.now()}-${Math.random().toString(16).slice(2)}`);

/* ---------- Tiny helpers ---------- */
const expectsYesNo = (text = "") =>
  /\b(yes\/no|\(yes\/no\)|yes\s*\/\s*no)\b/i.test(text);

const expectsPrice = (text = "") =>
  /(unit price|provide a unit price|price.*number only)/i.test(text);

const extractSuggestedPrice = (text = "") => {
  // finds first currency-styled number e.g., £1,299.99 or 1299.99
  const m = text.match(
    /(?:£|\$|€)?\s?(\d{1,3}(?:[,\d]{3})*(?:\.\d{1,2})?|\d+(?:\.\d{1,2})?)/
  );
  return m ? m[0].trim() : "";
};

// Normalize pdf_url from backend (path, full URL, or sometimes PO id)
// Returns a fully-qualified URL pointing at your FastAPI service.
function normalizePdfHref(pdf_url, po_number) {
  if (!pdf_url && po_number) {
    // Build from PO number if only that was returned
    return `${API_BASE}/po/${encodeURIComponent(po_number)}/pdf`;
  }

  if (!pdf_url) return null;

  // If backend mistakenly returned just the PO id as pdf_url
  if (!/^https?:\/\//i.test(pdf_url) && !pdf_url.startsWith("/")) {
    // if it looks like a PO id, build the API route
    if (/^PO[-_]/i.test(pdf_url) || /^[A-Za-z]+-?\d+/.test(pdf_url)) {
      return `${API_BASE}/po/${encodeURIComponent(pdf_url)}/pdf`;
    }
  }

  // If it's already absolute (https://...)
  if (/^https?:\/\//i.test(pdf_url)) return pdf_url;

  // It's a server path like /po/PO-123/pdf
  return `${API_BASE}${pdf_url}`;
}

const formatKbd = (key) => (
  <kbd className="px-1 py-0.5 rounded bg-white/10 border border-white/20">
    {key}
  </kbd>
);

/* ---------- Message bubble ---------- */
function MessageBubble({ msg }) {
  const role = ROLES[msg.role] || ROLES.bot;
  const time = useMemo(
    () =>
      new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
    []
  );

  // Highlight a PO number if present
  const poNumber =
    typeof msg.content === "string"
      ? (msg.content.match(/\bPO[:\s#-]*([A-Za-z0-9\-]+)\b/) || [])[1]
      : null;

  // Role-specific typography to maintain contrast
  const proseClass =
    msg.role === "bot"
      ? "prose prose-slate prose-p:my-2 prose-code:px-1 prose-code:py-0.5 prose-code:bg-black/10 prose-code:rounded"
      : "prose prose-invert prose-p:my-2 prose-code:px-1 prose-code:py-0.5 prose-code:bg-black/10 prose-code:rounded";

  return (
    <div className={`w-full flex ${role.align} gap-3`}>
      {/* Left avatar (bot) */}
      {msg.role !== "user" && (
        <div className="hidden sm:flex items-start">
          <div className="size-9 grid place-items-center rounded-xl bg-white/80 text-slate-900 shadow-sm">
            <span className="text-sm">{role.emoji}</span>
          </div>
        </div>
      )}

      {/* Bubble */}
      <div
        className={`max-w-[82%] sm:max-w-[78%] px-4 py-3 ${role.bubble} ${role.corner} shadow-[0_12px_30px_rgba(2,6,23,0.18)] border border-white/20`}
      >
        <div className="text-[11px] opacity-70 mb-1">
          {role.name} · {time}
        </div>

        {/* Optional PO tag */}
        {poNumber && msg.role === "bot" && (
          <div className="mb-2 inline-flex items-center gap-2 text-xs px-2 py-1 rounded-lg bg-emerald-500/15 text-emerald-900 border border-emerald-400/40">
            <span>PO</span>
            <code className="font-semibold">{poNumber}</code>
            <button
              onClick={() => navigator.clipboard?.writeText(poNumber)}
              className="ml-1 text-emerald-900/80 hover:text-emerald-900 underline"
              title="Copy PO number"
            >
              Copy
            </button>
          </div>
        )}

        <div className={proseClass}>
          <ReactMarkdown
            remarkPlugins={[remarkGfm]}
            disallowedElements={["script"]}
            components={{
              a: ({ node, ...props }) => (
                <a {...props} target="_blank" rel="noreferrer" />
              ),
            }}
          >
            {String(msg.content || "")}
          </ReactMarkdown>
        </div>
      </div>

      {/* Right avatar (user) */}
      {msg.role === "user" && (
        <div className="hidden sm:flex items-start">
          <div className="size-9 grid place-items-center rounded-xl bg-gradient-to-br from-indigo-500 to-fuchsia-500 text-white shadow-sm">
            <span className="text-sm">{role.emoji}</span>
          </div>
        </div>
      )}
    </div>
  );
}

/* ---------- Suggestions (chips under Send) ---------- */
const SUGGESTIONS = [
  "Create a Purchase Order for 15 Lenovo ThinkPads",
  "What are the approval rules for > £5000?",
  "Please approve 100 APPLE Macbook Pro",
  "Summarize today's PO activity",
];
const AUTO_SEND_ON_SUGGESTION = true;

/* ---------- Approval bar (HITL) ---------- */
function ApprovalBar({ prompt, onDecision, isLoading }) {
  const yesno = expectsYesNo(prompt);
  const priceAsk = expectsPrice(prompt);
  const suggested = extractSuggestedPrice(prompt);
  const [price, setPrice] = useState("");

  useEffect(() => {
    if (priceAsk && suggested) setPrice(suggested.replace(/[^\d.]/g, ""));
  }, [prompt]); // eslint-disable-line

  return (
    <div className="sticky bottom-0 z-20 -mx-4 sm:mx-0 mb-2 px-4 sm:px-0">
      <div className="rounded-2xl border border-amber-400/40 bg-amber-300/15 backdrop-blur p-3 sm:p-4 text-amber-900 shadow-[0_12px_30px_rgba(251,191,36,0.18)]">
        <div className="text-base sm:text-[15px] font-bold text-rose-800 mb-2 flex items-center gap-2">
          <span className="size-6 grid place-items-center rounded-lg bg-rose-100 text-rose-700">
            ⚠️
          </span>
          <span className="uppercase tracking-wide">Approval Required</span>
        </div>
        <div className="text-sm whitespace-pre-wrap mb-3">
          <div className="px-3 py-2 rounded-lg bg-rose-50 text-rose-800 border border-rose-300 shadow-sm">
            {prompt}
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          {yesno && (
            <>
              <button
                type="button"
                disabled={isLoading}
                onClick={() => onDecision("yes")}
                className="px-3 py-1.5 rounded-full bg-emerald-600 text-white hover:bg-emerald-500 shadow-sm disabled:opacity-60"
              >
                Approve
              </button>
              <button
                type="button"
                disabled={isLoading}
                onClick={() => onDecision("no")}
                className="px-3 py-1.5 rounded-full bg-rose-600 text-white hover:bg-rose-500 shadow-sm disabled:opacity-60"
              >
                Decline
              </button>
            </>
          )}

          {priceAsk && (
            <>
              <input
                inputMode="decimal"
                pattern="[0-9]*"
                value={price}
                onChange={(e) =>
                  setPrice(e.target.value.replace(/[^\d.]/g, ""))
                }
                placeholder="Enter unit price"
                className="px-3 py-1.5 rounded-xl border border-amber-500/30 bg-white/70 text-slate-900 focus:outline-none focus:ring-2 focus:ring-amber-400/50"
              />
              <button
                type="button"
                disabled={isLoading || !price}
                onClick={() => onDecision(price)}
                className="px-3 py-1.5 rounded-full bg-indigo-600 text-white hover:bg-indigo-500 shadow-sm disabled:opacity-60"
              >
                Submit Price
              </button>
              {suggested && (
                <button
                  type="button"
                  disabled={isLoading}
                  onClick={() => onDecision(suggested)}
                  className="px-3 py-1.5 rounded-full bg-white/20 border border-white/30 text-slate-900 hover:bg-white/30"
                  title={`Use ${suggested}`}
                >
                  Use {suggested}
                </button>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
}

/* ---------- App ---------- */
function App() {
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState([]);
  const [threadId, setThreadId] = useState(null);
  const [approvalPrompt, setApprovalPrompt] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const bottomRef = useRef(null);
  const inputRef = useRef(null);

  /* restore persisted state (helps when replying to an interrupt after refresh) */
  useEffect(() => {
    const t = localStorage.getItem("poThreadId");
    const ap = localStorage.getItem("poApprovalPrompt");
    if (t) setThreadId(t);
    if (ap) setApprovalPrompt(ap);
  }, []);

  /* persist thread + approval prompt */
  useEffect(() => {
    if (threadId) localStorage.setItem("poThreadId", String(threadId));
    else localStorage.removeItem("poThreadId");
  }, [threadId]);

  useEffect(() => {
    if (approvalPrompt) localStorage.setItem("poApprovalPrompt", approvalPrompt);
    else localStorage.removeItem("poApprovalPrompt");
  }, [approvalPrompt]);

  /* scroll to bottom on updates */
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isLoading, approvalPrompt]);

  /* auto-grow textarea */
  useEffect(() => {
    const el = inputRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = Math.min(el.scrollHeight, 240) + "px";
  }, [input]);

  const pushMsg = (role, content) =>
    setMessages((prev) => [...prev, { id: uid(), role, content }]);

  const sendDecision = async (decisionText) => {
    setIsLoading(true);
    try {
      const res = await axios.post(`${API_BASE}/agnt/chat`, {
        decision: String(decisionText),
        conversation_id: threadId,
      });

      const { status, message, prompt, pdf_url, po_number } = res.data || {};

      if (status === "APPROVAL_REQUIRED" && prompt) {
        pushMsg("bot", prompt);
        setApprovalPrompt(prompt);
      } else {
        // Build a final message + PDF link
        let finalMsg = message || "Action completed.";
        const href = normalizePdfHref(pdf_url, po_number);
        if (href) {
          finalMsg += `\n\n[View PO PDF](${href})`;
        }
        // Add PO number if not already present in the text
        if (po_number && !/\bPO[:\s#-]*[A-Za-z0-9\-]+\b/.test(finalMsg)) {
          finalMsg = `PO: ${po_number}\n\n${finalMsg}`;
        }

        pushMsg("bot", finalMsg);
        setApprovalPrompt("");
      }
    } catch (err) {
      console.error(err);
      const detail =
        err?.response?.data?.error || err.message || "Unknown error";
      pushMsg("bot", `❌ Something went wrong.\n\n${detail}`);
    } finally {
      setIsLoading(false);
    }
  };

  const handleSend = async (maybeText) => {
    const text = (typeof maybeText === "string" ? maybeText : input).trim();
    if (!text) return;

    // If user typed a bare number but we’re not in an approval step, nudge them.
    if (!approvalPrompt && /^[0-9]+(\.[0-9]+)?$/.test(text)) {
      pushMsg(
        "bot",
        "That looks like a price reply. If you're answering an approval question, please continue in the same thread. Otherwise, try a full request like **“Buy 3 Lenovo ThinkPads at £199”**."
      );
      setInput("");
      return;
    }

    setInput("");
    setIsLoading(true);
    pushMsg("user", text);

    try {
      if (approvalPrompt) {
        await sendDecision(text);
        return;
      }

      // normal question flow
      const res = await axios.post(`${API_BASE}/agnt/chat`, {
        Question: text,
        ...(threadId ? { conversation_id: threadId } : {}),
      });

      const {
        status,
        thread_id,
        prompt,
        message,
        pdf_url,
        po_number,
      } = res.data || {};
      if (thread_id) setThreadId(thread_id);

      if (status === "APPROVAL_REQUIRED") {
        pushMsg("bot", prompt || "Please approve this.");
        setApprovalPrompt(prompt || "");
      } else {
        let finalMsg = message || "No response received.";
        const href = normalizePdfHref(pdf_url, po_number);
        if (href) {
          finalMsg += `\n\n[View PO PDF](${href})`;
        }
        if (po_number && !/\bPO[:\s#-]*[A-Za-z0-9\-]+\b/.test(finalMsg)) {
          finalMsg = `PO: ${po_number}\n\n${finalMsg}`;
        }

        pushMsg("bot", finalMsg);
      }
    } catch (err) {
      console.error(err);
      const detail =
        err?.response?.data?.error || err.message || "Unknown error";
      pushMsg("bot", `❌ Something went wrong.\n\n${detail}`);
    } finally {
      setIsLoading(false);
    }
  };

  const onSuggestionClick = async (text) => {
    setInput(text);
    if (AUTO_SEND_ON_SUGGESTION) {
      await new Promise((r) => setTimeout(r, 60));
      handleSend(text);
    }
  };

  const handleKeyDown = (e) => {
    if (isLoading) return;
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const resetThread = () => {
    setThreadId(null);
    setMessages([]);
    setApprovalPrompt("");
    setInput("");
    localStorage.removeItem("poThreadId");
    localStorage.removeItem("poApprovalPrompt");
  };

  return (
    <div className="min-h-screen w-full text-white font-poppins bg-[radial-gradient(1200px_800px_at_-10%_-20%,rgba(120,113,108,0.18)_0%,transparent_60%),radial-gradient(900px_600px_at_110%_0%,rgba(68,64,60,0.18)_0%,transparent_55%),linear-gradient(180deg,#0b0f12_0%,#0d1114_55%,#0c1012_100%)]">
      <div className="max-w-6xl mx-auto py-8 sm:py-10 px-4 sm:px-6 lg:px-8 flex flex-col h-full">
        {/* Header */}
        <div className="sticky top-0 z-10 -mx-4 sm:mx-0 px-4 sm:px-0 pb-4 mb-6 border-b border-white/10 bg-transparent">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="size-9 grid place-items-center rounded-2xl bg-gradient-to-br from-cyan-500 to-indigo-500 shadow-[0_8px_24px_rgba(56,189,248,0.35)]">
                <span className="text-lg">🧾</span>
              </div>
              <div>
                <div className="text-xl font-semibold">PO Assistant</div>
                <div className="text-xs text-white/60"></div>
              </div>
            </div>
            <button
              onClick={resetThread}
              type="button"
              className="px-3 py-2 rounded-xl border border-white/15 bg-white/10 hover:bg-white/20 active:scale-[.98] transition"
            >
              Reset
            </button>
          </div>
        </div>

        {/* Chat shell */}
        <div
          className="rounded-3xl border border-white/10 bg-white/5 backdrop-blur-xl shadow-[0_24px_60px_rgba(2,6,23,0.45)] flex flex-col min-h-[70vh]"
          aria-busy={isLoading}
        >
          {/* Messages */}
          <div className="flex-1 overflow-y-auto px-4 sm:px-6 py-6 space-y-4">
            {messages.length === 0 ? (
              <div className="h-full grid place-items-center">
                <div className="w-full max-w-3xl text-center space-y-6">
                  <h2 className="text-2xl sm:text-3xl font-semibold">
                    <span className="bg-gradient-to-r from-indigo-300 via-cyan-200 to-fuchsia-300 bg-clip-text text-transparent">
                      Chat Mode
                    </span>
                  </h2>

                  <p className="text-white/70 max-w-xl mx-auto">
                    Create purchase orders, check stock & prices, and follow
                    approval policy — all in one chat.
                  </p>

                  {/* Feature cards */}
                  <div className="grid sm:grid-cols-3 gap-3 text-left">
                    <div className="rounded-2xl border border-white/15 bg-white/8 p-4 backdrop-blur">
                      <div className="text-lg mb-1">🧾 Create POs</div>
                      <p className="text-sm text-white/70">
                        “Create a Purchase Order for <b>15 Lenovo ThinkPads</b>”
                      </p>
                    </div>
                    <div className="rounded-2xl border border-white/15 bg-white/8 p-4 backdrop-blur">
                      <div className="text-lg mb-1">✅ Approvals</div>
                      <p className="text-sm text-white/70">
                        “What are the approval rules for <b>&gt; £5000</b>?”
                      </p>
                    </div>
                    <div className="rounded-2xl border border-white/15 bg-white/8 p-4 backdrop-blur">
                      <div className="text-lg mb-1">🔎 Vendor Pricing</div>
                      <p className="text-sm text-white/70">
                        “Find web price for <b>Apple MacBook Pro</b>”
                      </p>
                    </div>
                  </div>

                  {/* Quick suggestions */}
                  <div className="flex flex-wrap justify-center gap-2"></div>

                  <div className="text-[11px] text-white/60">
                    Tip: Press {formatKbd("Enter")} to send · {formatKbd("Shift")}{" "}
                    for new line
                  </div>
                </div>
              </div>
            ) : (
              <>
                {messages.map((msg) => (
                  <MessageBubble key={msg.id} msg={msg} />
                ))}

                {isLoading && (
                  <div className="flex items-center gap-2 text-white/80 text-sm">
                    <span className="size-2 rounded-full bg-white/80 animate-pulse" />
                    Thinking…
                  </div>
                )}

                <div ref={bottomRef} />
                {DEBUG && threadId && (
                  <p className="text-xs text-white/50 mt-2">thread: {threadId}</p>
                )}
              </>
            )}
          </div>

          {/* Approval bar (sticky over composer) */}
          {approvalPrompt && (
            <ApprovalBar
              prompt={approvalPrompt}
              onDecision={(d) => sendDecision(d)}
              isLoading={isLoading}
            />
          )}

          {/* Composer */}
          <div className="border-t border-white/10 p-3 sm:p-4">
            <div className="flex items-end gap-2">
              <textarea
                ref={inputRef}
                rows={1}
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder={
                  approvalPrompt ? "Type your response…" : "Ask Anything..."
                }
                aria-label={approvalPrompt ? "Approval response" : "Message input"}
                className="flex-1 leading-relaxed px-4 py-3 rounded-2xl border border-white/15 bg-white/10 placeholder-white/60 text-white focus:outline-none focus:ring-2 focus:ring-cyan-300/40 resize-none"
              />
              <button
                onClick={() => handleSend()}
                type="button"
                className="h-11 px-5 sm:px-6 rounded-full font-semibold bg-indigo-600 hover:bg-indigo-500 text-white shadow-[0_10px_30px_rgba(99,102,241,0.35)] active:scale-[.98] transition disabled:opacity-60"
                disabled={isLoading}
                aria-disabled={isLoading}
              >
                Send 🚀
              </button>
            </div>

            {/* Suggestions under Send */}
            {!approvalPrompt && (
              <>
                <div className="mt-3 flex flex-wrap items-center gap-2">
                  {SUGGESTIONS.map((s) => (
                    <button
                      key={s}
                      type="button"
                      onClick={() => onSuggestionClick(s)}
                      className="text-xs sm:text-sm px-3 py-1.5 rounded-full bg-white/15 border border-white/20 text-white hover:bg-white/25 transition shadow-sm"
                      title={s}
                    >
                      {s}
                    </button>
                  ))}
                </div>
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
