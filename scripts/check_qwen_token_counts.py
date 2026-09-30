"""Development-only compatibility check; sends three tiny model requests.

Run: python -m scripts.check_qwen_token_counts
"""
import os

from openai import OpenAI

from human_eval.qwen_token_counter import QwenTokenCounter


def main():
    client = OpenAI(base_url=os.environ.get("DOF_AGENT_BASE_URL", "http://192.168.1.117:8888/v1"), api_key="local")
    counter = QwenTokenCounter(reasoning_effort="low")
    tool = {"type": "function", "function": {
        "name": "read_chunks", "description": "Lee evidencia del DOF.",
        "parameters": {"type": "object", "properties": {
            "chunk_ids": {"type": "array", "items": {"type": "integer"}}},
            "required": ["chunk_ids"]}}}
    base = [{"role": "system", "content": "Responde usando evidencia del DOF."},
            {"role": "user", "content": "¿Qué establece el decreto?"}]
    history = [*base, {"role": "assistant", "content": "",
        "tool_calls": [{"id": "call_1", "type": "function", "function": {
            "name": "read_chunks", "arguments": '{"chunk_ids":[123]}'}}]},
        {"role": "tool", "tool_call_id": "call_1", "content": "Artículo 1. El decreto entra en vigor mañana."},
        {"role": "user", "content": "¿Cuándo entra en vigor?"}]
    for name, messages, tools in [("simple", base, []), ("tools", base, [tool]), ("history", history, [tool])]:
        expected = counter.count_messages(messages, tools)
        response = client.chat.completions.create(
            model=os.environ.get("DOF_AGENT_MODEL", "Qwen3.8-Flash-Next"),
            messages=messages, max_tokens=1,
            extra_body={"chat_template_kwargs": {"enable_thinking": True, "reasoning_effort": "low"}},
            **({"tools": tools, "tool_choice": "auto"} if tools else {}))
        actual = response.usage.prompt_tokens
        print(f"{name}: local={expected}, server={actual}")
        assert expected == actual, f"Tokenizer/template mismatch: {name}"


if __name__ == "__main__":
    main()
