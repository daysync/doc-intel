"""Provider-neutral request and response types.

The pipeline builds an ``LLMRequest`` and gets back an ``LLMResponse[T]``. Nothing in here
mentions a provider; adapters translate to and from their SDK's shapes.
"""

import hashlib
from datetime import datetime
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

ImageMime = Literal["image/jpeg", "image/png", "image/webp"]


class TextPart(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["text"] = "text"
    text: str


class ImagePart(BaseModel):
    """Raw image bytes. HEIC is converted to JPEG before it gets here (Stage 2)."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["image"] = "image"
    data: bytes
    mime: ImageMime


Part = TextPart | ImagePart


class Message(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["user", "assistant"]
    parts: list[Part]


class LLMRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str = Field(
        description="Provider-specific model id, e.g. claude-opus-5, gpt-5.5, qwen2.5vl:3b"
    )
    system: str
    messages: list[Message]
    max_tokens: int = 4096
    temperature: float | None = Field(
        default=None,
        description="None = provider default. Adapters drop it where the provider rejects it.",
    )

    def fingerprint(self, schema: dict[str, object]) -> str:
        """Stable SHA-256 over everything that influences the answer.

        Images are hashed by content so the key stays short and fixtures never embed bytes.
        Used by the recorded adapter to find the fixture for a request.
        """
        digest = hashlib.sha256()
        digest.update(self.model.encode())
        digest.update(b"\x00")
        digest.update(self.system.encode())
        digest.update(b"\x00")
        digest.update(str(self.max_tokens).encode())
        digest.update(b"\x00")
        digest.update(str(self.temperature).encode())
        for message in self.messages:
            digest.update(b"\x00" + message.role.encode())
            for part in message.parts:
                if isinstance(part, TextPart):
                    digest.update(b"\x00text:" + part.text.encode())
                else:
                    digest.update(b"\x00image:" + hashlib.sha256(part.data).hexdigest().encode())
        digest.update(b"\x00schema:" + _canonical_json(schema).encode())
        return digest.hexdigest()


class RawCompletion(BaseModel):
    """What an adapter hands back: the model's text plus token counts. Nothing parsed yet."""

    model_config = ConfigDict(extra="forbid")

    text: str
    input_tokens: int
    output_tokens: int
    cached_input_tokens: int = 0
    request_id: str | None = None


class CallRecord(BaseModel):
    """One line in the cost ledger. Every call produces exactly one."""

    model_config = ConfigDict(extra="forbid")

    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    cached_input_tokens: int
    cost_usd: Decimal
    latency_ms: int
    started_at: datetime
    request_id: str | None = None
    fixture_key: str | None = Field(default=None, description="Set when served from a fixture")


class LLMResponse[T](BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    output: T
    raw_text: str
    record: CallRecord


def _canonical_json(value: object) -> str:
    import json

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
