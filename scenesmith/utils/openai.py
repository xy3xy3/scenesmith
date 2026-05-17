import base64
import os

from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np

from agents import RunConfig
from agents.models.interface import ModelProvider
from agents.models.openai_provider import OpenAIProvider
from openai import AsyncOpenAI, OpenAI
from PIL import Image


def encode_image_to_base64(image: np.ndarray | str | Path) -> str:
    """Encodes an image to a base64 string.

    Args:
        image: Either a numpy array of shape (H, W, 3) in RGB format, a path string,
            or a Path object to an image file.

    Returns:
        str: The base64 encoded image string.
    """
    if isinstance(image, (str, Path)):
        # Read image directly from path.
        with Image.open(image) as img:
            # Convert to RGB in case it's not.
            img = img.convert("RGB")
            # Save to bytes.
            buffer = BytesIO()
            img.save(buffer, format="JPEG")
            return base64.b64encode(buffer.getvalue()).decode("utf-8")
    else:
        # Convert numpy array to PIL Image.
        img = Image.fromarray(image)
        buffer = BytesIO()
        img.save(buffer, format="JPEG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")


def get_openai_base_url(cfg: Any | None = None) -> str | None:
    """Resolve a custom OpenAI-compatible base URL from config or environment."""
    config_base_url = None
    if cfg is not None:
        openai_cfg = getattr(cfg, "openai", None)
        if openai_cfg is not None:
            config_base_url = getattr(openai_cfg, "base_url", None)

    if config_base_url:
        return str(config_base_url)

    return os.environ.get("OPENAI_BASE_URL") or os.environ.get("OPENAI_API_BASE")


def get_openai_use_responses(cfg: Any | None = None) -> bool:
    """Resolve whether to use the Responses API for OpenAI-compatible providers."""
    config_use_responses = None
    if cfg is not None:
        openai_cfg = getattr(cfg, "openai", None)
        if openai_cfg is not None:
            config_use_responses = getattr(openai_cfg, "use_responses", None)

    if config_use_responses is not None:
        return bool(config_use_responses)

    env_value = os.environ.get("OPENAI_USE_RESPONSES")
    if env_value is not None:
        return env_value.strip().lower() in {"1", "true", "yes", "on"}

    return True


def get_openai_client_kwargs(cfg: Any | None = None) -> dict[str, Any]:
    """Return keyword args shared by OpenAI sync/async clients."""
    kwargs: dict[str, Any] = {}
    base_url = get_openai_base_url(cfg)
    if base_url:
        kwargs["base_url"] = base_url
    return kwargs


def create_openai_client(cfg: Any | None = None, **overrides: Any) -> OpenAI:
    """Create a sync OpenAI client honoring project config/env overrides."""
    kwargs = get_openai_client_kwargs(cfg)
    kwargs.update(overrides)
    return OpenAI(**kwargs)


def create_async_openai_client(cfg: Any | None = None, **overrides: Any) -> AsyncOpenAI:
    """Create an async OpenAI client honoring project config/env overrides."""
    kwargs = get_openai_client_kwargs(cfg)
    kwargs.update(overrides)
    return AsyncOpenAI(**kwargs)


def create_openai_model_provider(
    cfg: Any | None = None, *, use_responses: bool | None = None
) -> ModelProvider:
    """Create an OpenAI Agents model provider honoring custom base_url."""
    kwargs = get_openai_client_kwargs(cfg)
    kwargs["use_responses"] = (
        use_responses if use_responses is not None else get_openai_use_responses(cfg)
    )
    return OpenAIProvider(**kwargs)


def create_openai_run_config(
    cfg: Any | None = None, **run_config_kwargs: Any
) -> RunConfig:
    """Create a RunConfig that uses the project's OpenAI provider settings."""
    return RunConfig(
        model_provider=create_openai_model_provider(cfg),
        **run_config_kwargs,
    )
