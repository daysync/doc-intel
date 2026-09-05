from doc_intel.llm.types import ImagePart, LLMRequest, Message, TextPart


def _request(**overrides: object) -> LLMRequest:
    base: dict[str, object] = {
        "model": "m",
        "system": "s",
        "messages": [Message(role="user", parts=[TextPart(text="hello")])],
    }
    base.update(overrides)
    return LLMRequest.model_validate(base)


def test_fingerprint_is_stable_for_equal_requests() -> None:
    assert _request().fingerprint({"a": 1}) == _request().fingerprint({"a": 1})


def test_fingerprint_ignores_schema_key_order() -> None:
    assert _request().fingerprint({"a": 1, "b": 2}) == _request().fingerprint({"b": 2, "a": 1})


def test_fingerprint_changes_when_anything_meaningful_changes() -> None:
    base = _request().fingerprint({})
    assert _request(model="other").fingerprint({}) != base
    assert _request(system="other").fingerprint({}) != base
    assert _request(max_tokens=1).fingerprint({}) != base
    assert _request().fingerprint({"changed": True}) != base


def test_fingerprint_hashes_image_bytes_not_their_size() -> None:
    def with_image(data: bytes) -> str:
        message = Message(
            role="user", parts=[ImagePart(data=data, mime="image/png"), TextPart(text="x")]
        )
        return _request(messages=[message]).fingerprint({})

    assert with_image(b"aaaa") != with_image(b"bbbb")
    assert with_image(b"aaaa") == with_image(b"aaaa")
