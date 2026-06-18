import json
from pathlib import Path

import streamlit as st

from src.chatbot import answer_query
from src.config import (
    HUNDRED_CHECKPOINTS_PATH,
    PERSONA_PATH,
    RETRIEVAL_DOCUMENTS_PATH,
    TOPIC_CHECKPOINTS_PATH,
)


st.set_page_config(
    page_title="Conversation RAG Persona Chatbot",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
      .block-container { padding-top: 1.4rem; padding-bottom: 2rem; }
      [data-testid="stMetricValue"] { font-size: 1.4rem; }
      .evidence-box {
        border: 1px solid #d9dee7;
        border-radius: 6px;
        padding: 0.85rem;
        background: #f7f9fc;
        margin-bottom: 0.65rem;
      }
      .small-muted { color: #5f6b7a; font-size: 0.88rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_json(path: str):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def processed_files_ready() -> bool:
    required = [
        TOPIC_CHECKPOINTS_PATH,
        HUNDRED_CHECKPOINTS_PATH,
        PERSONA_PATH,
        RETRIEVAL_DOCUMENTS_PATH,
    ]
    return all(path.exists() for path in required)


def show_retrieval_results(title: str, results: list[dict]) -> None:
    st.subheader(title)
    if not results:
        st.info("No matching evidence found.")
        return

    for doc in results:
        with st.expander(
            f"{doc['title']} | messages {doc['start_message_id']}-{doc['end_message_id']} | score {doc['score']}",
            expanded=False,
        ):
            st.caption(doc["type"])
            st.write(doc["text"])
            st.json(doc.get("metadata", {}), expanded=False)


def show_persona_card(speaker: str, persona: dict) -> None:
    st.subheader(speaker)
    style = persona["communication_style"]
    metrics = st.columns(4)
    metrics[0].metric("Messages", style["message_count"])
    metrics[1].metric("Avg Words", style["average_content_words"])
    metrics[2].metric("Question Rate", style["question_rate"])
    metrics[3].metric("Exclamation Rate", style["exclamation_rate"])
    st.write(", ".join(style["style_notes"]))

    for category in ["personal_facts", "habits", "preferences", "personality_traits"]:
        with st.expander(category.replace("_", " ").title(), expanded=category == "habits"):
            for item in persona[category][:12]:
                evidence = item["evidence"][0]
                st.markdown(f"**{item['claim']}**  ")
                st.caption(
                    f"{item['confidence']} confidence | seen {item['count']} times | "
                    f"message {evidence['message_id']}"
                )
                st.write(evidence["text"])


def checkpoint_table(checkpoints: list[dict], page_size: int, key_prefix: str) -> None:
    max_page = max((len(checkpoints) - 1) // page_size, 0)
    page = st.number_input(
        "Page",
        min_value=1,
        max_value=max_page + 1,
        value=1,
        step=1,
        key=f"{key_prefix}_page",
    )
    start = (page - 1) * page_size
    end = start + page_size
    rows = []
    for checkpoint in checkpoints[start:end]:
        metadata = checkpoint.get("metadata", {})
        rows.append(
            {
                "id": checkpoint.get("topic_id", checkpoint.get("checkpoint_id")),
                "start": checkpoint["start_message_id"],
                "end": checkpoint["end_message_id"],
                "messages": checkpoint.get("message_count", metadata.get("message_count", "")),
                "keywords": ", ".join(checkpoint.get("keywords", [])[:6]),
                "summary": checkpoint.get("summary", checkpoint.get("text", "")),
            }
        )
    st.dataframe(rows, width="stretch", hide_index=True)


if not processed_files_ready():
    st.error("Processed files are missing. Run `python preprocess.py` once, then restart the app.")
    st.stop()


topic_checkpoints = load_json(str(TOPIC_CHECKPOINTS_PATH))
hundred_checkpoints = load_json(str(HUNDRED_CHECKPOINTS_PATH))
persona = load_json(str(PERSONA_PATH))
retrieval_documents = load_json(str(RETRIEVAL_DOCUMENTS_PATH))

st.title("Conversation RAG Persona Chatbot")

with st.sidebar:
    st.header("Project Status")
    st.metric("Topic Checkpoints", len(topic_checkpoints))
    st.metric("100-Message Checkpoints", len(hundred_checkpoints))
    st.metric("Retrieval Documents", len(retrieval_documents))
    st.metric("Speakers", len(persona["speakers"]))
    speaker_filter = st.selectbox("Speaker Focus", ["All", "User 1", "User 2"])

tabs = st.tabs(["Chatbot", "Persona", "Topic Checkpoints", "100-Message Checkpoints", "Evidence Corpus"])

with tabs[0]:
    example_questions = [
        "What kind of person is this user?",
        "What are User 1 habits?",
        "How does User 2 talk?",
        "What did the user say about moving to Portland?",
    ]

    if "query_text" not in st.session_state:
        st.session_state["query_text"] = ""

    st.caption("Ask any question, or use one of the examples to fill the input.")
    example_cols = st.columns(4)
    for index, example in enumerate(example_questions):
        if example_cols[index].button(example, key=f"example_{index}", use_container_width=True):
            st.session_state["query_text"] = example

    with st.form("chat_form"):
        query = st.text_input(
            "Question",
            key="query_text",
            placeholder="Type your own question here, for example: What does User 1 like to do for fun?",
        )
        submitted = st.form_submit_button("Ask", type="primary")

    if submitted and query.strip():
        focused_query = query
        if speaker_filter != "All" and speaker_filter.lower() not in query.lower():
            focused_query = f"{query} {speaker_filter}"
        with st.spinner("Retrieving summaries, chunks, and persona evidence..."):
            result = answer_query(focused_query)
        st.session_state["last_result"] = result
    elif submitted:
        st.warning("Type a question first.")

    if "last_result" in st.session_state:
        result = st.session_state["last_result"]
        st.subheader("Answer")
        st.markdown(result["answer"])
        evidence_tabs = st.tabs(["Topic Summaries", "Message Chunks", "100-Message Summaries", "Raw Evidence Text"])
        with evidence_tabs[0]:
            show_retrieval_results("Retrieved Topic Summaries", result["topic_results"])
        with evidence_tabs[1]:
            show_retrieval_results("Retrieved Message Chunks", result["message_chunk_results"])
        with evidence_tabs[2]:
            show_retrieval_results("Retrieved 100-Message Summaries", result["hundred_message_results"])
        with evidence_tabs[3]:
            st.text(result["evidence_text"])

with tabs[1]:
    for speaker, speaker_persona in persona["speakers"].items():
        show_persona_card(speaker, speaker_persona)

with tabs[2]:
    st.subheader("Topic Checkpoints")
    checkpoint_table(topic_checkpoints, page_size=25, key_prefix="topic")

with tabs[3]:
    st.subheader("100-Message Checkpoints")
    checkpoint_table(hundred_checkpoints, page_size=25, key_prefix="hundred")

with tabs[4]:
    counts = {}
    for document in retrieval_documents:
        counts[document["type"]] = counts.get(document["type"], 0) + 1
    st.json(counts)
    checkpoint_table(retrieval_documents[:500], page_size=25, key_prefix="corpus")
