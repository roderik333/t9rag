"""Mistral?."""

from collections.abc import Generator

from mistralai import Mistral


class MistralLLM:
    def __init__(
        self,
        model_name: str,
        api_key: str,
        timeout: int,
        context_window: int,
        verbose: bool,
        max_tokens: int,
        temperature: float,
        top_p: int,
        num_gpu: int,
    ):
        super().__init__()
        self.model_name = model_name
        self.api_key = api_key
        self.timeout = timeout
        self.context_window = context_window
        self.verbose = verbose
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.num_gpu = num_gpu
        self.client = Mistral(api_key=self.api_key)

    def stream_complete(self, prompt: str) -> Generator[str, None, None]:
        for response in self.client.chat.stream(
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=1,
            n=1,
            stream=True,
        ):
            if response.data.choices[0].delta.content == "[DONE]":
                break
            yield response.data.choices[0].delta.content


def initialize_mistral_llm(
    model_name: str,
    api_key: str,
    timeout: int,
    context_window: int,
    verbose: bool,
    max_tokens: int,
    temperature: float,
    top_p: int,
    num_gpu: int,
) -> MistralLLM:
    return MistralLLM(model_name, api_key, timeout, context_window, verbose, max_tokens, temperature, top_p, num_gpu)


def mistral_stream_complete(llm: MistralLLM, prompt: str) -> Generator[str, None, None]:
    return llm.stream_complete(prompt)
