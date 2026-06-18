import json

from src.checkpoint_builder import (
    build_hundred_message_checkpoints,
    build_topic_checkpoints,
    checkpoint_preview,
)
from src.config import (
    DATASET_PATH,
    HUNDRED_CHECKPOINTS_PATH,
    MESSAGES_PATH,
    PERSONA_PATH,
    PROCESSED_DIR,
    RETRIEVAL_DOCUMENTS_PATH,
    TOPIC_CHECKPOINTS_PATH,
)
from src.data_loader import load_messages
from src.persona_extractor import extract_persona
from src.retriever import build_message_chunks, build_retrieval_documents


def write_json(path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    messages = load_messages(DATASET_PATH)
    topic_checkpoints = build_topic_checkpoints(messages)
    hundred_checkpoints = build_hundred_message_checkpoints(messages)
    persona = extract_persona(messages)
    message_chunks = build_message_chunks(messages)
    retrieval_documents = build_retrieval_documents(topic_checkpoints, hundred_checkpoints, message_chunks)

    write_json(MESSAGES_PATH, messages)
    write_json(TOPIC_CHECKPOINTS_PATH, topic_checkpoints)
    write_json(HUNDRED_CHECKPOINTS_PATH, hundred_checkpoints)
    write_json(PERSONA_PATH, persona)
    write_json(RETRIEVAL_DOCUMENTS_PATH, retrieval_documents)

    print(f"Parsed messages: {len(messages)}")
    print(f"Topic checkpoints: {len(topic_checkpoints)}")
    print(f"100-message checkpoints: {len(hundred_checkpoints)}")
    print(f"Persona speakers: {', '.join(persona['speakers'])}")
    print(f"Retrieval documents: {len(retrieval_documents)}")
    print(f"Message chunks: {len(message_chunks)}")
    print("\nTopic preview:")
    print(checkpoint_preview(topic_checkpoints))
    print("\n100-message preview:")
    print(checkpoint_preview(hundred_checkpoints, limit=3))
    print("\nPersona preview:")
    for speaker, speaker_persona in persona["speakers"].items():
        facts = speaker_persona["personal_facts"][:2]
        habits = speaker_persona["habits"][:2]
        print(f"{speaker}: {len(speaker_persona['personal_facts'])} facts, {len(habits)} shown habits")
        for item in facts + habits:
            print(f"- {item['claim']} ({item['confidence']}, evidence: message {item['evidence'][0]['message_id']})")


if __name__ == "__main__":
    main()
