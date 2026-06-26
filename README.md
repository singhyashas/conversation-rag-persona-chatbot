# Conversation RAG Persona Chatbot

This project builds a local-first RAG chatbot over a CSV of conversations. Each row in the dataset is treated as one chronological conversation, and every `User 1:` / `User 2:` line is parsed into an individual message before any checkpointing or retrieval is done.

## Features

- Chronological message parsing from `conversations.csv`
- Topic checkpoints when the conversation topic changes
- Independent 100-message checkpoints
- Local BM25-style retrieval over topic summaries and raw message chunks
- Evidence-backed persona extraction in JSON
- Streamlit chatbot UI with retrieved evidence panels
- No external LLM API dependency

## Project Structure

```text
.
├── app.py
├── conversations.csv
├── preprocess.py
├── requirements.txt
├── src/
│   ├── checkpoint_builder.py
│   ├── chatbot.py
│   ├── config.py
│   ├── data_loader.py
│   ├── persona_extractor.py
│   ├── retriever.py
│   ├── summarizer.py
│   └── text_utils.py
└── data/processed/
    ├── hundred_message_checkpoints.json
    ├── messages.json
    ├── persona.json
    ├── retrieval_documents.json
    └── topic_checkpoints.json
```

## How To Run Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Generate processed files:

```bash
python preprocess.py
```

Run the chatbot app:

```bash
streamlit run app.py
```

You can also query the backend directly:

```bash
python -m src.chatbot "What are User 1 habits?"
python -m src.chatbot "What did the user say about moving to Portland?"
python -m src.chatbot "How does User 2 talk?"
```

## How Topic Changes Are Detected

The system does not treat a full conversation row as one topic. It first parses each row into chronological messages, then processes those messages in order.

A new topic checkpoint is created when:

- the conversation row changes, or
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

These checkpoints are used as broader chronological memory blocks during retrieval and inspection.

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

Retrieval uses a local BM25-style lexical scorer implemented in `src/retriever.py`. This keeps the project lightweight and avoids relying on external APIs.

## How Persona Is Built

Persona extraction is implemented in `src/persona_extractor.py`.

The system extracts:

- habits
- personal facts
- preferences
- personality traits
- communication style

Facts, habits, preferences, and traits are included only when there is actual message evidence. Each claim stores evidence with message id, conversation id, speaker, and original text.

Example:

```json
{
  "claim": "software engineer",
  "confidence": "high",
  "evidence": [
    {
      "message_id": 89,
      "conversation_id": 5,
      "speaker": "User 1",
      "text": "I'm a software engineer."
    }
  ]
}
```

Communication style is measured from observable statistics, including average message length, question rate, exclamation rate, short-message rate, first-person rate, emoji count, and top repeated terms.

## Current Processed Output

The current dataset run produced:

- `191,592` parsed messages
- `20,492` topic checkpoints
- `1,916` 100-message checkpoints
- `35,181` retrieval documents
- persona JSON for `User 1` and `User 2`


Hosted app URL:

```text
https://kastack-m6qtwwxrpdlou94ex7hgrp.streamlit.app/
```

## Demo

Loom video:

```text

```

]

## Notes

The project intentionally uses local logic and lightweight retrieval. It does not depend on ChatGPT or external LLM APIs, which keeps the implementation explainable and reproducible.
