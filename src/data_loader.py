import csv
import re
from pathlib import Path

from .text_utils import normalize_space


MESSAGE_RE = re.compile(
    r"(User\s+\d+):\s*(.*?)(?=\nUser\s+\d+:\s*|\Z)",
    re.IGNORECASE | re.DOTALL,
)


def parse_conversation(raw_text: str) -> list[dict]:
    messages = []
    for match in MESSAGE_RE.finditer(raw_text.strip()):
        speaker = normalize_space(match.group(1)).title()
        text = normalize_space(match.group(2))
        if text:
            messages.append({"speaker": speaker, "text": text})
    return messages


def load_messages(csv_path: Path) -> list[dict]:
    parsed_messages = []
    global_message_id = 1

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        for conversation_id, row in enumerate(reader, start=1):
            if not row:
                continue
            conversation_text = row[0]
            for message_index, message in enumerate(parse_conversation(conversation_text), start=1):
                parsed_messages.append(
                    {
                        "global_message_id": global_message_id,
                        "conversation_id": conversation_id,
                        "message_index_in_conversation": message_index,
                        "speaker": message["speaker"],
                        "text": message["text"],
                    }
                )
                global_message_id += 1

    return parsed_messages
