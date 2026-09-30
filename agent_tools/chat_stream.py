"""Collect OpenAI chat deltas while forwarding text to a live UI."""
from openai.types.chat import ChatCompletion


def collect_chat_stream(client, kwargs, on_delta):
    message = {"role": "assistant", "content": "", "reasoning_content": ""}
    calls = {}
    usage = None
    finish = None
    response_id = "stream"
    with client.chat.completions.create(
        **kwargs, stream=True, stream_options={"include_usage": True}
    ) as stream:
        for chunk in stream:
            response_id = chunk.id
            if chunk.usage is not None:
                usage = chunk.usage.model_dump(exclude_none=True)
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            finish = choice.finish_reason or finish
            delta = choice.delta.model_dump(exclude_none=True)
            for field in ("content", "reasoning_content"):
                text = delta.get(field) or ""
                if text:
                    message[field] += text
                    on_delta(field, text)
            for call in delta.get("tool_calls", []):
                item = calls.setdefault(call["index"], {"id": "", "type": "function", "function": {"name": "", "arguments": ""}})
                if call.get("id"):
                    item["id"] = call["id"]
                for field in ("name", "arguments"):
                    item["function"][field] += call.get("function", {}).get(field) or ""
    if calls:
        message["tool_calls"] = [calls[index] for index in sorted(calls)]
    on_delta("done", "")
    return ChatCompletion.model_validate({
        "id": response_id, "object": "chat.completion", "created": 0,
        "model": kwargs["model"], "usage": usage,
        "choices": [{"index": 0, "message": message, "finish_reason": finish or "stop"}],
    })
