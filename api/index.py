from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse

from src.chatbot import answer_query
from src.config import (
    HUNDRED_CHECKPOINTS_PATH,
    PERSONA_PATH,
    RETRIEVAL_DOCUMENTS_PATH,
    TOPIC_CHECKPOINTS_PATH,
)


app = FastAPI(title="Conversation RAG Persona Chatbot")


def processed_files_ready() -> bool:
    return all(
        path.exists()
        for path in (
            TOPIC_CHECKPOINTS_PATH,
            HUNDRED_CHECKPOINTS_PATH,
            PERSONA_PATH,
            RETRIEVAL_DOCUMENTS_PATH,
        )
    )


def compact_docs(documents: list[dict], max_chars: int = 700) -> list[dict]:
    compact = []
    for doc in documents:
        text = " ".join(doc.get("text", "").split())
        if len(text) > max_chars:
            text = text[: max_chars - 3].rstrip() + "..."
        compact.append(
            {
                "title": doc.get("title"),
                "type": doc.get("type"),
                "score": doc.get("score"),
                "start_message_id": doc.get("start_message_id"),
                "end_message_id": doc.get("end_message_id"),
                "text": text,
            }
        )
    return compact


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    return """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Conversation RAG Persona Chatbot</title>
    <style>
      :root {
        color-scheme: light;
        --ink: #18202a;
        --muted: #667085;
        --line: #d8dee8;
        --soft: #f5f7fb;
        --accent: #0f766e;
      }
      body {
        margin: 0;
        font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        color: var(--ink);
        background: #ffffff;
      }
      main {
        width: min(1120px, calc(100% - 32px));
        margin: 36px auto 56px;
      }
      h1 {
        margin: 0 0 8px;
        font-size: clamp(30px, 5vw, 52px);
        letter-spacing: 0;
      }
      p {
        line-height: 1.55;
      }
      .lede {
        color: var(--muted);
        max-width: 780px;
        margin-bottom: 24px;
      }
      .panel {
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 18px;
        background: var(--soft);
      }
      .row {
        display: flex;
        gap: 10px;
      }
      input {
        flex: 1;
        min-width: 0;
        border: 1px solid var(--line);
        border-radius: 6px;
        padding: 13px 14px;
        font-size: 16px;
        background: #fff;
      }
      button {
        border: 0;
        border-radius: 6px;
        padding: 0 18px;
        color: #fff;
        background: var(--accent);
        font-weight: 700;
        cursor: pointer;
      }
      .examples {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin: 12px 0 0;
      }
      .examples button {
        background: #e7f3f1;
        color: #0b5c56;
        padding: 9px 11px;
        font-weight: 650;
      }
      #answer, #evidence {
        white-space: pre-wrap;
      }
      .section {
        margin-top: 20px;
      }
      .doc {
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 14px;
        margin-top: 10px;
        background: #fff;
      }
      .doc strong {
        display: block;
        margin-bottom: 6px;
      }
      .meta {
        color: var(--muted);
        font-size: 13px;
        margin-bottom: 8px;
      }
      @media (max-width: 720px) {
        .row {
          flex-direction: column;
        }
        button {
          min-height: 44px;
        }
      }
    </style>
  </head>
  <body>
    <main>
      <h1>Conversation RAG Persona Chatbot</h1>
      <p class="lede">
        A lightweight Vercel interface over the local RAG backend. It retrieves topic summaries,
        raw message chunks, 100-message checkpoints, and evidence-backed persona data.
      </p>
      <section class="panel">
        <div class="row">
          <input id="query" placeholder="Ask any question, e.g. What are User 1 habits?" />
          <button onclick="ask()">Ask</button>
        </div>
        <div class="examples">
          <button onclick="fillQuestion('What kind of person is this user?')">What kind of person is this user?</button>
          <button onclick="fillQuestion('What are User 1 habits?')">What are User 1 habits?</button>
          <button onclick="fillQuestion('How does User 2 talk?')">How does User 2 talk?</button>
          <button onclick="fillQuestion('What did the user say about moving to Portland?')">Moving to Portland</button>
        </div>
      </section>
      <section class="section">
        <h2>Answer</h2>
        <div id="answer" class="panel">Ask a question to begin.</div>
      </section>
      <section class="section">
        <h2>Retrieved Evidence</h2>
        <div id="evidence"></div>
      </section>
    </main>
    <script>
      function fillQuestion(value) {
        document.getElementById("query").value = value;
      }

      function renderDocs(title, docs) {
        if (!docs || !docs.length) return "";
        return `<h3>${title}</h3>` + docs.map(doc => `
          <div class="doc">
            <strong>${doc.title}</strong>
            <div class="meta">${doc.type} | messages ${doc.start_message_id}-${doc.end_message_id} | score ${doc.score}</div>
            <div>${doc.text}</div>
          </div>
        `).join("");
      }

      async function ask() {
        const query = document.getElementById("query").value.trim();
        if (!query) return;
        document.getElementById("answer").textContent = "Retrieving evidence...";
        document.getElementById("evidence").innerHTML = "";
        const response = await fetch("/api/ask", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ query })
        });
        const data = await response.json();
        if (!response.ok) {
          document.getElementById("answer").textContent = data.error || "Something went wrong.";
          return;
        }
        document.getElementById("answer").textContent = data.answer;
        document.getElementById("evidence").innerHTML =
          renderDocs("Topic Summaries", data.topic_results) +
          renderDocs("Message Chunks", data.message_chunk_results) +
          renderDocs("100-Message Summaries", data.hundred_message_results);
      }
    </script>
  </body>
</html>
"""


@app.get("/api/status")
def status() -> dict:
    return {
        "app": "Conversation RAG Persona Chatbot",
        "processed_files_ready": processed_files_ready(),
    }


@app.post("/api/ask")
async def ask(request: Request) -> JSONResponse:
    if not processed_files_ready():
        return JSONResponse(
            {"error": "Processed files are missing. Run python preprocess.py before deploying."},
            status_code=500,
        )

    payload = await request.json()
    query = str(payload.get("query", "")).strip()
    if not query:
        return JSONResponse({"error": "Question is required."}, status_code=400)

    result = answer_query(query)
    return JSONResponse(
        {
            "query": query,
            "answer": result["answer"],
            "topic_results": compact_docs(result["topic_results"]),
            "message_chunk_results": compact_docs(result["message_chunk_results"]),
            "hundred_message_results": compact_docs(result["hundred_message_results"]),
        }
    )
