from dataclasses import dataclass


@dataclass(frozen=True)
class PromptContext:
    system_prompt: str
    user_prompt: str


class ContextSeparator:
    @staticmethod
    def build_context(system_prompt: str, user_prompt: str) -> PromptContext:
        if not system_prompt:
            raise ValueError("System prompt must not be empty")

        if not isinstance(system_prompt, str):
            raise TypeError("System prompt must be string")

        if not isinstance(user_prompt, str):
            raise TypeError("User prompt must be string")

        return PromptContext(
            system_prompt=system_prompt,
            user_prompt=user_prompt
        )

