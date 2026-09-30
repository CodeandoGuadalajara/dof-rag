"""Text-only TensorFold v0.3.6.3 request counting for our chat adapter."""
import json
from copy import deepcopy
from typing import Any

from agent_tools.agent import OpenAIChatCompletionsBackend

TOKENIZER_REPO = "Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP"
TOKENIZER_REVISION = "dadefa8066e3be900a0d148d0f5a2f4eb1cf6534"


class QwenTokenCounter:
    def __init__(self, *, enable_thinking: bool = True, reasoning_effort: str | None = None):
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            TOKENIZER_REPO, revision=TOKENIZER_REVISION
        )
        self.enable_thinking = enable_thinking
        self.reasoning_effort = reasoning_effort

    def count_messages(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> int:
        messages = deepcopy(messages)
        for message in messages:
            if not isinstance(message.get("content", ""), (str, type(None))):
                raise ValueError("only text messages are supported")
            if message.get("content") is None:
                message["content"] = ""
            for call in message.get("tool_calls", []):
                arguments = call["function"].get("arguments")
                if isinstance(arguments, str):
                    parsed = json.loads(arguments)
                    if isinstance(parsed, dict):
                        call["function"]["arguments"] = parsed
        kwargs = dict(add_generation_prompt=True, tokenize=False,
                      enable_thinking=self.enable_thinking,
                      thinking_mode="thinking" if self.enable_thinking else "chat")
        if tools:
            kwargs["tools"] = tools
        if self.enable_thinking and self.reasoning_effort:
            kwargs["reasoning_effort"] = self.reasoning_effort
        rendered = self.tokenizer.apply_chat_template(messages, **kwargs)
        return len(self.tokenizer.encode(rendered))

    def __call__(self, *, input_items, tools, instructions) -> int:
        return self.count_messages(
            OpenAIChatCompletionsBackend._messages(input_items, instructions),
            OpenAIChatCompletionsBackend._chat_tools(tools),
        )
