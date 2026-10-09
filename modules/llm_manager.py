from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

import requests


class LLMError(RuntimeError):
    """Ollama is unavailable or returned an invalid model response."""


@dataclass(frozen=True)
class ModelInfo:
    name: str
    digest: str
    size_bytes: int
    parameter_size: str


@dataclass(frozen=True)
class GenerationResult:
    model: str
    response: str
    duration_seconds: float


@dataclass(frozen=True)
class ComparisonResult:
    model: str
    response: str | None
    duration_seconds: float | None
    error: str | None


class OllamaManager:
    def __init__(self, base_url: str, timeout_seconds: int = 120) -> None:
        self.base_url = base_url.rstrip("/")
        parsed_url = urlsplit(self.base_url)
        if parsed_url.scheme not in {"http", "https"} or not parsed_url.hostname:
            raise ValueError("OLLAMA_URL must be an absolute HTTP or HTTPS URL.")
        if parsed_url.username or parsed_url.password:
            raise ValueError("Do not put credentials in OLLAMA_URL.")
        self.timeout_seconds = timeout_seconds
        self.session = requests.Session()

    def list_models(self) -> list[ModelInfo]:
        try:
            response = self.session.get(
                f"{self.base_url}/api/tags", timeout=min(self.timeout_seconds, 10)
            )
            response.raise_for_status()
            payload = response.json()
            models = payload.get("models")
            if not isinstance(models, list):
                raise LLMError("Ollama returned a malformed model list.")
            return [
                ModelInfo(
                    name=str(model["name"]),
                    digest=str(model.get("digest", "")),
                    size_bytes=int(model.get("size", 0)),
                    parameter_size=str(model.get("details", {}).get("parameter_size", "")),
                )
                for model in models
                if isinstance(model, dict) and model.get("name")
            ]
        except LLMError:
            raise
        except requests.RequestException as error:
            raise LLMError(
                "Cannot connect to Ollama at "
                f"{self.base_url}. Start Ollama and confirm its local API is available."
            ) from error
        except (ValueError, TypeError, KeyError) as error:
            raise LLMError("Ollama returned malformed model metadata.") from error

    def generate(
        self,
        model: str,
        prompt: str,
        temperature: float = 0.2,
        max_tokens: int = 1200,
    ) -> GenerationResult:
        if not 0 <= temperature <= 2:
            raise ValueError("Temperature must be between 0 and 2.")
        if not 1 <= max_tokens <= 8192:
            raise ValueError("Maximum output tokens must be between 1 and 8192.")
        available = {item.name for item in self.list_models()}
        if model not in available:
            raise LLMError(
                f"Model '{model}' is not installed in Ollama. Select an available "
                "model or install it yourself with Ollama; the app will not pull it."
            )

        started = time.perf_counter()
        try:
            response = self.session.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens,
                    },
                },
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            payload: Any = response.json()
            answer = payload.get("response") if isinstance(payload, dict) else None
            if not isinstance(answer, str) or not answer.strip():
                raise LLMError("Ollama returned an empty or malformed response.")
            elapsed = time.perf_counter() - started
            duration_ns = payload.get("total_duration")
            if isinstance(duration_ns, (int, float)) and duration_ns >= 0:
                elapsed = duration_ns / 1_000_000_000
            return GenerationResult(model, answer.strip(), elapsed)
        except LLMError:
            raise
        except requests.Timeout as error:
            raise LLMError(
                f"Model '{model}' timed out after {self.timeout_seconds} seconds."
            ) from error
        except requests.RequestException as error:
            raise LLMError(f"Ollama request for '{model}' failed: {error}") from error
        except ValueError as error:
            raise LLMError("Ollama returned a response that was not valid JSON.") from error

    def compare(
        self, first_model: str, second_model: str, prompt: str, **options: Any
    ) -> list[ComparisonResult]:
        available = self.list_models()
        by_name = {item.name: item for item in available}
        if first_model not in by_name or second_model not in by_name:
            missing = [
                name for name in (first_model, second_model) if name not in by_name
            ]
            raise LLMError(
                f"Selected model(s) are unavailable: {', '.join(missing)}."
            )
        first = by_name[first_model]
        second = by_name[second_model]
        if first_model == second_model or (
            first.digest and first.digest == second.digest
        ):
            raise LLMError(
                "Choose two different installed models. These tags resolve to the "
                "same underlying model digest and would duplicate the comparison."
            )

        outcomes: list[ComparisonResult] = []
        for model in (first_model, second_model):
            try:
                result = self.generate(model, prompt, **options)
                outcomes.append(
                    ComparisonResult(
                        model, result.response, result.duration_seconds, None
                    )
                )
            except LLMError as error:
                outcomes.append(ComparisonResult(model, None, None, str(error)))
        return outcomes
