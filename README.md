# Conversation RAG Persona Chatbot

A FastAPI-based RAG chatbot for analyzing conversation history. The project processes conversations chronologically, creates topic and 100-message checkpoints, retrieves relevant evidence, and extracts an evidence-backed user persona.

The system is designed to be explainable and local-first. It uses deterministic parsing, checkpointing, summarization, and BM25-style retrieval implemented in Python instead of depending on external LLM APIs.

## Links

- GitHub Repository: `https://github.com/singhyashas/conversation-rag-persona-chatbot`
- Live Vercel App: `https://singhyashas-conversation-rag-person.vercel.app/`

## Features

- Chronological message parsing from `conversations.csv`
- Topic checkpoints whenever the active topic changes
- Independent 100-message checkpoints
- BM25-style retrieval over topic summaries, message chunks, and 100-message summaries
- Persona extraction with evidence-backed habits, facts, traits, and communication style
- FastAPI web UI and JSON API for chatbot queries
- Vercel deployment support
- Dockerized FastAPI deployment

## Project Structure

```text
.
|-- api/
|   `-- index.py
|-- conversations.csv
|-- Dockerfile
|-- preprocess.py
|-- requirements.txt
|-- vercel.json
|-- src/
|   |-- checkpoint_builder.py
|   |-- chatbot.py
|   |-- config.py
|   |-- data_loader.py
|   |-- persona_extractor.py
|   |-- retriever.py
|   |-- summarizer.py
|   `-- text_utils.py
`-- data/processed/
    |-- hundred_message_checkpoints.json
    |-- messages.json
    |-- persona.json
    |-- retrieval_documents.json
    `-- topic_checkpoints.json
```

## Requirements

- Python 3.12 recommended
- Dependencies from `requirements.txt`

```bash
pip install -r requirements.txt
```

## Run Locally

Generate processed files if needed:

```bash
python preprocess.py
```

Start the FastAPI app:

```bash
uvicorn api.index:app --reload
```

Open:

```text
http://localhost:8000
```

Useful API endpoints:

```text
GET  /api/status
POST /api/ask
```

Example API request:

```bash
curl -X POST http://localhost:8000/api/ask \
  -H "Content-Type: application/json" \
  -d "{\"query\": \"What are User 1 habits?\"}"
```

You can also query the backend from the command line:

```bash
python -m src.chatbot "What are User 1 habits?"
python -m src.chatbot "What did the user say about moving to Portland?"
python -m src.chatbot "How does User 2 talk?"
```

## Docker

Build the image:

```bash
docker build -t conversation-rag-persona-chatbot .
```

Run the container:

```bash
docker run --rm -p 8000:8000 conversation-rag-persona-chatbot
```

Open:

```text
http://localhost:8000
```

The Docker image runs the FastAPI app with Uvicorn. The `.dockerignore` file excludes development files, screenshots, the raw CSV, and `messages.json`; the runtime keeps the processed retrieval documents, topic checkpoints, 100-message checkpoints, and persona file.

## Vercel Deployment

This project deploys on Vercel through `api/index.py`, which exports a top-level FastAPI `app` object.

Vercel settings:

```text
Framework Preset: Other
Build Command: leave empty
Output Directory: leave empty
Install Command: pip install -r requirements.txt
```

The `vercel.json` file rewrites all routes to the FastAPI entrypoint:

```json
{
  "rewrites": [
    {
      "source": "/(.*)",
      "destination": "/api/index"
    }
  ]
}
```

Before deploying, make sure these processed files are committed:

```text
data/processed/topic_checkpoints.json
data/processed/hundred_message_checkpoints.json
data/processed/persona.json
data/processed/retrieval_documents.json
```

Vercel app URL:

```text
https://singhyashas-conversation-rag-person.vercel.app/
```

## How Topic Changes Are Detected

The system does not treat a full conversation row as one topic. It first parses each row into chronological messages, then processes those messages in order.

A new topic checkpoint is created when:

- the conversation row changes,
- a topic-shift phrase appears, such as `what about`, `what do you do`, `for fun`, or `do you have any`, or
- lexical similarity between the current message and the recent active topic window drops below the configured threshold.

Each topic checkpoint stores:

- topic id
- start and end message ids
- message count
- split reason
- keywords
- summary for that segment only

Example output shape:

```json
{
  "topic_id": 1,
  "start_message_id": 1,
  "end_message_id": 8,
  "message_count": 8,
  "keywords": ["moving", "portland", "city"],
  "summary": "I'm excited to be moving to a new city soon..."
}
```

## How 100-Message Checkpoints Work

The 100-message checkpoints are independent from topic checkpoints. After messages are parsed chronologically, the system creates one summary for messages `1-100`, another for `101-200`, and so on until the dataset ends.

These checkpoints provide broader chronological memory blocks during retrieval.

## How Retrieval Works

The retrieval corpus contains three document types:

- `topic_summary`: summaries of topic-based segments
- `message_chunk`: overlapping raw message windows
- `hundred_message_summary`: fixed 100-message summaries

When a question is asked, the chatbot:

1. retrieves relevant topic summaries,
2. retrieves relevant raw message chunks,
3. retrieves a small number of 100-message summaries for broader context,
4. combines the retrieved evidence with persona data when the query is persona-related.

Retrieval uses a local BM25-style lexical scorer implemented in `src/retriever.py`.

## How Persona Is Built

Persona extraction is implemented in `src/persona_extractor.py`.

The system extracts:

- habits
- personal facts
- preferences
- personality traits
- communication style

Claims are included only when there is message evidence. Each claim stores evidence with message id, conversation id, speaker, and original text.

Communication style is measured from observable statistics, including average message length, question rate, exclamation rate, short-message rate, first-person rate, emoji count, and top repeated terms.

## Current Processed Output

The current dataset run produced:

- `191,592` parsed messages
- `20,492` topic checkpoints
- `1,916` 100-message checkpoints
- `35,181` retrieval documents
- persona JSON for `User 1` and `User 2`

## Demo

Loom video:

```text
TODO: paste demo video URL here
```

Screenshots are available in the `screenshots/` folder.

## Notes

The project intentionally uses local logic and lightweight retrieval. It does not depend on external hosted LLM APIs, which keeps the implementation explainable and reproducible.
