import json
import re
from functools import lru_cache
from pathlib import Path

from .config import PERSONA_PATH, RETRIEVAL_DOCUMENTS_PATH
from .retriever import BM25Retriever, load_documents


PERSONA_QUERY_TERMS = {
    "person",
    "persona",
    "habit",
    "habits",
    "personality",
    "trait",
    "traits",
    "talk",
    "communicate",
    "communication",
    "style",
    "facts",
    "fact",
    "kind",
}


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


@lru_cache(maxsize=1)
def load_knowledge_base() -> tuple[BM25Retriever, dict]:
    documents = load_documents(RETRIEVAL_DOCUMENTS_PATH)
    persona = load_json(PERSONA_PATH)
    return BM25Retriever(documents), persona


def query_mentions_persona(query: str) -> bool:
    lowered = query.lower()
    return any(term in lowered for term in PERSONA_QUERY_TERMS)


def target_speakers(query: str, persona: dict) -> list[str]:
    lowered = query.lower()
    speakers = list(persona["speakers"].keys())
    if "user 1" in lowered:
        return ["User 1"]
    if "user 2" in lowered:
        return ["User 2"]
    return speakers


def select_persona_categories(query: str) -> list[str]:
    lowered = query.lower()
    categories = []
    if any(term in lowered for term in ("habit", "routine", "usually", "often")):
        categories.append("habits")
    if any(term in lowered for term in ("fact", "personal", "relationship", "family", "event")):
        categories.append("personal_facts")
    if any(term in lowered for term in ("person", "personality", "trait", "kind")):
        categories.append("personality_traits")
    if any(term in lowered for term in ("talk", "communicat", "style", "tone", "message")):
        categories.append("communication_style")
    if not categories:
        categories = ["personal_facts", "habits", "personality_traits", "communication_style"]
    return categories


def format_doc_result(document: dict, max_chars: int = 600) -> str:
    text = re.sub(r"\s+", " ", document["text"]).strip()
    if len(text) > max_chars:
        text = text[: max_chars - 3].rstrip() + "..."
    return (
        f"{document['title']} [{document['type']}, messages "
        f"{document['start_message_id']}-{document['end_message_id']}, score {document['score']}]: {text}"
    )


def format_persona_item(item: dict) -> str:
    evidence = item["evidence"][0]
    return (
        f"{item['claim']} ({item['confidence']}, seen {item['count']} times; "
        f"evidence message {evidence['message_id']}: \"{evidence['text']}\")"
    )


def persona_answer(query: str, persona: dict) -> str:
    speakers = target_speakers(query, persona)
    categories = select_persona_categories(query)
    sections = []

    for speaker in speakers:
        speaker_persona = persona["speakers"][speaker]
        lines = [f"{speaker}:"]
        for category in categories:
            if category == "communication_style":
                style = speaker_persona["communication_style"]
                lines.append(
                    "Communication style: "
                    + ", ".join(style["style_notes"])
                    + (
                        f" (question rate {style['question_rate']}, "
                        f"exclamation rate {style['exclamation_rate']}, "
                        f"avg content words {style['average_content_words']})."
                    )
                )
            else:
                items = speaker_persona[category][:5]
                readable_category = category.replace("_", " ")
                if items:
                    lines.append(f"{readable_category}:")
                    lines.extend(f"- {format_persona_item(item)}" for item in items)
        sections.append("\n".join(lines))

    return "\n\n".join(sections)


def evidence_block(topic_results: list[dict], chunk_results: list[dict], hundred_results: list[dict]) -> str:
    lines = ["Retrieved topic summaries:"]
    lines.extend(f"- {format_doc_result(doc)}" for doc in topic_results)
    lines.append("\nRetrieved message chunks:")
    lines.extend(f"- {format_doc_result(doc)}" for doc in chunk_results)
    if hundred_results:
        lines.append("\nRetrieved 100-message checkpoints:")
        lines.extend(f"- {format_doc_result(doc)}" for doc in hundred_results)
    return "\n".join(lines)


def general_rag_answer(query: str, topic_results: list[dict], chunk_results: list[dict]) -> str:
    if not topic_results and not chunk_results:
        return "I could not find strong matching evidence in the indexed conversations."

    answer_lines = [
        "I found the most relevant evidence in the retrieved topic summaries and raw message chunks.",
    ]
    if topic_results:
        top_topic = topic_results[0]
        answer_lines.append(
            f"The strongest topic-level match is {top_topic['title']} "
            f"(messages {top_topic['start_message_id']}-{top_topic['end_message_id']}), "
            f"which says: {top_topic['text'].splitlines()[0].replace('Topic summary: ', '')}"
        )
    if chunk_results:
        top_chunk = chunk_results[0]
        snippet = re.sub(r"\s+", " ", top_chunk["text"]).strip()
        if len(snippet) > 350:
            snippet = snippet[:347].rstrip() + "..."
        answer_lines.append(
            f"The strongest raw-message match is messages "
            f"{top_chunk['start_message_id']}-{top_chunk['end_message_id']}: {snippet}"
        )
    return "\n".join(answer_lines)


def answer_query(query: str) -> dict:
    retriever, persona = load_knowledge_base()
    topic_results = retriever.search(query, top_k=5, doc_type="topic_summary")
    chunk_results = retriever.search(query, top_k=5, doc_type="message_chunk")
    hundred_results = retriever.search(query, top_k=2, doc_type="hundred_message_summary")

    parts = []
    if query_mentions_persona(query):
        parts.append(persona_answer(query, persona))
    parts.append(general_rag_answer(query, topic_results, chunk_results))

    return {
        "query": query,
        "answer": "\n\n".join(part for part in parts if part),
        "topic_results": topic_results,
        "message_chunk_results": chunk_results,
        "hundred_message_results": hundred_results,
        "evidence_text": evidence_block(topic_results, chunk_results, hundred_results),
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Ask the local RAG chatbot a question.")
    parser.add_argument("query", help="Question to ask")
    args = parser.parse_args()

    result = answer_query(args.query)
    print(result["answer"])
    print("\n--- Evidence ---")
    print(result["evidence_text"])


if __name__ == "__main__":
    main()
