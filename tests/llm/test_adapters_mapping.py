"""The provider adapters split into a pure mapping function and a thin network call.

These tests cover the mapping, which is where the shape bugs live. The network calls are
covered by recorded fixtures and the smoke run.
"""

import base64
from typing import Any, cast

from doc_intel.llm.anthropic import to_anthropic_messages
from doc_intel.llm.ollama import to_ollama_messages
from doc_intel.llm.openai import json_schema_format, to_openai_input
from doc_intel.llm.types import ImagePart, Message, TextPart

PNG = b"\x89PNG\r\n\x1a\nfake"
MESSAGES = [
    Message(role="user", parts=[ImagePart(data=PNG, mime="image/png"), TextPart(text="Read it.")]),
    Message(role="assistant", parts=[TextPart(text='{"ok": true}')]),
]
B64 = base64.standard_b64encode(PNG).decode()


def test_anthropic_mapping_puts_images_as_base64_blocks() -> None:
    mapped = cast(list[dict[str, Any]], to_anthropic_messages(MESSAGES))
    assert mapped[0]["role"] == "user"
    blocks = list(mapped[0]["content"])
    assert blocks[0] == {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": B64},
    }
    assert blocks[1] == {"type": "text", "text": "Read it."}
    assert mapped[1]["role"] == "assistant"


def test_openai_mapping_uses_data_urls_and_strict_schema() -> None:
    mapped = cast(list[dict[str, Any]], to_openai_input(MESSAGES))
    assert mapped[0]["role"] == "user"
    content = list(mapped[0]["content"])
    assert content[0]["type"] == "input_image"
    assert content[0]["image_url"] == f"data:image/png;base64,{B64}"
    assert content[1] == {"type": "input_text", "text": "Read it."}
    fmt = cast(dict[str, Any], json_schema_format({"type": "object"})["format"])
    assert fmt["type"] == "json_schema"
    assert fmt["strict"] is True


def test_ollama_mapping_prepends_system_and_attaches_image_bytes() -> None:
    mapped = to_ollama_messages("be terse", MESSAGES)
    assert mapped[0].role == "system"
    assert mapped[0].content == "be terse"
    assert mapped[1].role == "user"
    assert mapped[1].content == "Read it."
    assert mapped[1].images is not None
    assert mapped[1].images[0].value == PNG
    assert mapped[2].images is None
