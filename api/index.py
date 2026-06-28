import json
from functools import lru_cache

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


@lru_cache(maxsize=1)
def load_processed_data() -> dict:
    with TOPIC_CHECKPOINTS_PATH.open("r", encoding="utf-8") as handle:
        topic_checkpoints = json.load(handle)
    with HUNDRED_CHECKPOINTS_PATH.open("r", encoding="utf-8") as handle:
        hundred_checkpoints = json.load(handle)
    with PERSONA_PATH.open("r", encoding="utf-8") as handle:
        persona = json.load(handle)
    with RETRIEVAL_DOCUMENTS_PATH.open("r", encoding="utf-8") as handle:
        retrieval_documents = json.load(handle)
    return {
        "topic_checkpoints": topic_checkpoints,
        "hundred_checkpoints": hundred_checkpoints,
        "persona": persona,
        "retrieval_documents": retrieval_documents,
    }


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


def compact_checkpoint(checkpoint: dict, max_chars: int = 420) -> dict:
    summary = " ".join(checkpoint.get("summary", checkpoint.get("text", "")).split())
    if len(summary) > max_chars:
        summary = summary[: max_chars - 3].rstrip() + "..."
    return {
        "id": checkpoint.get("topic_id", checkpoint.get("checkpoint_id", checkpoint.get("doc_id"))),
        "title": checkpoint.get("title"),
        "type": checkpoint.get("type"),
        "start_message_id": checkpoint.get("start_message_id"),
        "end_message_id": checkpoint.get("end_message_id"),
        "message_count": checkpoint.get("message_count", checkpoint.get("metadata", {}).get("message_count")),
        "keywords": checkpoint.get("keywords", [])[:8],
        "summary": summary,
    }


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
        --accent-soft: #e7f3f1;
      }
      * {
        box-sizing: border-box;
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
      .status-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 10px;
        margin: 22px 0 18px;
      }
      .metric {
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 14px;
        background: #fff;
      }
      .metric span {
        display: block;
        color: var(--muted);
        font-size: 13px;
      }
      .metric strong {
        display: block;
        margin-top: 5px;
        font-size: 24px;
      }
      .tabs {
        display: flex;
        gap: 8px;
        flex-wrap: wrap;
        border-bottom: 1px solid var(--line);
        margin: 10px 0 24px;
      }
      .tab {
        background: transparent;
        color: var(--ink);
        border-radius: 0;
        padding: 12px 10px;
        border-bottom: 3px solid transparent;
      }
      .tab.active {
        color: var(--accent);
        border-bottom-color: var(--accent);
      }
      .tab-panel {
        display: none;
      }
      .tab-panel.active {
        display: block;
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
        background: var(--accent-soft);
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
      .grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 12px;
      }
      .persona-card {
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 16px;
        background: #fff;
      }
      .claim {
        border-top: 1px solid var(--line);
        padding-top: 10px;
        margin-top: 10px;
      }
      .claim strong {
        display: block;
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
        .status-grid, .grid {
          grid-template-columns: 1fr;
        }
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
      <section class="status-grid" id="status"></section>
      <nav class="tabs">
        <button class="tab active" onclick="showTab('chat', this)">Chatbot</button>
        <button class="tab" onclick="showTab('persona', this)">Persona</button>
        <button class="tab" onclick="showTab('topics', this)">Topic Checkpoints</button>
        <button class="tab" onclick="showTab('hundreds', this)">100-Message Checkpoints</button>
        <button class="tab" onclick="showTab('corpus', this)">Evidence Corpus</button>
      </nav>
      <section id="chat" class="tab-panel active">
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
      </section>
      <section id="persona" class="tab-panel">
        <h2>Persona</h2>
        <div id="persona-content" class="grid"></div>
      </section>
      <section id="topics" class="tab-panel">
        <h2>Topic Checkpoints</h2>
        <div id="topic-content"></div>
      </section>
      <section id="hundreds" class="tab-panel">
        <h2>100-Message Checkpoints</h2>
        <div id="hundred-content"></div>
      </section>
      <section id="corpus" class="tab-panel">
        <h2>Evidence Corpus</h2>
        <div id="corpus-content"></div>
      </section>
    </main>
    <script>
      function showTab(id, button) {
        document.querySelectorAll(".tab-panel").forEach(panel => panel.classList.remove("active"));
        document.querySelectorAll(".tab").forEach(tab => tab.classList.remove("active"));
        document.getElementById(id).classList.add("active");
        button.classList.add("active");
      }

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

      function renderCheckpointDocs(docs) {
        return docs.map(doc => `
          <div class="doc">
            <strong>${doc.title || `${doc.type || "checkpoint"} ${doc.id}`}</strong>
            <div class="meta">messages ${doc.start_message_id}-${doc.end_message_id} | ${doc.message_count || ""} messages</div>
            <div>${doc.summary || doc.text || ""}</div>
            ${doc.keywords && doc.keywords.length ? `<div class="meta">keywords: ${doc.keywords.join(", ")}</div>` : ""}
          </div>
        `).join("");
      }

      function renderPersona(data) {
        const speakers = data.speakers || {};
        document.getElementById("persona-content").innerHTML = Object.entries(speakers).map(([speaker, persona]) => {
          const style = persona.communication_style || {};
          const claims = ["habits", "personal_facts", "personality_traits"].map(category => {
            const items = (persona[category] || []).slice(0, 5);
            if (!items.length) return "";
            return `<h3>${category.replaceAll("_", " ")}</h3>` + items.map(item => `
              <div class="claim">
                <strong>${item.claim}</strong>
                <div class="meta">${item.confidence} confidence | seen ${item.count} times</div>
                <div>${item.evidence && item.evidence[0] ? item.evidence[0].text : ""}</div>
              </div>
            `).join("");
          }).join("");
          return `
            <article class="persona-card">
              <h3>${speaker}</h3>
              <p>${(style.style_notes || []).join(", ")}</p>
              <div class="meta">messages ${style.message_count || 0} | avg words ${style.average_content_words || 0} | question rate ${style.question_rate || 0}</div>
              ${claims}
            </article>
          `;
        }).join("");
      }

      async function loadDashboard() {
        const response = await fetch("/api/dashboard");
        const data = await response.json();
        document.getElementById("status").innerHTML = `
          <div class="metric"><span>Topic Checkpoints</span><strong>${data.counts.topic_checkpoints}</strong></div>
          <div class="metric"><span>100-Message Checkpoints</span><strong>${data.counts.hundred_checkpoints}</strong></div>
          <div class="metric"><span>Retrieval Documents</span><strong>${data.counts.retrieval_documents}</strong></div>
          <div class="metric"><span>Speakers</span><strong>${data.counts.speakers}</strong></div>
        `;
        renderPersona(data.persona);
        document.getElementById("topic-content").innerHTML = renderCheckpointDocs(data.topic_checkpoints);
        document.getElementById("hundred-content").innerHTML = renderCheckpointDocs(data.hundred_checkpoints);
        document.getElementById("corpus-content").innerHTML = renderCheckpointDocs(data.retrieval_documents);
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

      loadDashboard();
    </script>
  </body>
</html>
"""


@app.get("/api/status")
def status() -> dict:
    counts = {}
    if processed_files_ready():
        data = load_processed_data()
        counts = {
            "topic_checkpoints": len(data["topic_checkpoints"]),
            "hundred_checkpoints": len(data["hundred_checkpoints"]),
            "retrieval_documents": len(data["retrieval_documents"]),
            "speakers": len(data["persona"].get("speakers", {})),
        }
    return {
        "app": "Conversation RAG Persona Chatbot",
        "processed_files_ready": processed_files_ready(),
        "counts": counts,
    }


@app.get("/api/dashboard")
def dashboard() -> JSONResponse:
    if not processed_files_ready():
        return JSONResponse(
            {"error": "Processed files are missing. Run python preprocess.py before deploying."},
            status_code=500,
        )
    data = load_processed_data()
    return JSONResponse(
        {
            "counts": {
                "topic_checkpoints": len(data["topic_checkpoints"]),
                "hundred_checkpoints": len(data["hundred_checkpoints"]),
                "retrieval_documents": len(data["retrieval_documents"]),
                "speakers": len(data["persona"].get("speakers", {})),
            },
            "persona": data["persona"],
            "topic_checkpoints": [
                compact_checkpoint(checkpoint) for checkpoint in data["topic_checkpoints"][:50]
            ],
            "hundred_checkpoints": [
                compact_checkpoint(checkpoint) for checkpoint in data["hundred_checkpoints"][:50]
            ],
            "retrieval_documents": [
                compact_checkpoint(document) for document in data["retrieval_documents"][:50]
            ],
        }
    )


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
