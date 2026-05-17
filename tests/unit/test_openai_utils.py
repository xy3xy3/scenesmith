import os
import unittest

from types import SimpleNamespace
from unittest.mock import patch

from scenesmith.utils.openai import (
    create_async_openai_client,
    create_openai_client,
    get_openai_base_url,
    get_openai_use_responses,
)


class TestOpenAIUtils(unittest.TestCase):
    def test_get_openai_base_url_prefers_cfg(self):
        cfg = SimpleNamespace(openai=SimpleNamespace(base_url="https://cfg.example/v1"))
        with patch.dict(os.environ, {"OPENAI_BASE_URL": "https://env.example/v1"}):
            self.assertEqual(
                get_openai_base_url(cfg),
                "https://cfg.example/v1",
            )

    def test_get_openai_base_url_falls_back_to_env_alias(self):
        with patch.dict(
            os.environ,
            {"OPENAI_API_BASE": "https://alias.example/v1"},
            clear=True,
        ):
            self.assertEqual(
                get_openai_base_url(),
                "https://alias.example/v1",
            )

    def test_get_openai_use_responses_prefers_cfg(self):
        cfg = SimpleNamespace(openai=SimpleNamespace(use_responses=False))
        with patch.dict(os.environ, {"OPENAI_USE_RESPONSES": "true"}):
            self.assertFalse(get_openai_use_responses(cfg))

    def test_get_openai_use_responses_reads_env(self):
        with patch.dict(os.environ, {"OPENAI_USE_RESPONSES": "false"}, clear=True):
            self.assertFalse(get_openai_use_responses())

    @patch("scenesmith.utils.openai.OpenAI")
    def test_create_openai_client_passes_base_url(self, mock_openai):
        cfg = SimpleNamespace(openai=SimpleNamespace(base_url="https://cfg.example/v1"))

        create_openai_client(cfg)

        mock_openai.assert_called_once_with(base_url="https://cfg.example/v1")

    @patch("scenesmith.utils.openai.AsyncOpenAI")
    def test_create_async_openai_client_passes_base_url(self, mock_async_openai):
        with patch.dict(
            os.environ,
            {"OPENAI_BASE_URL": "https://env.example/v1"},
            clear=True,
        ):
            create_async_openai_client()

        mock_async_openai.assert_called_once_with(base_url="https://env.example/v1")


if __name__ == "__main__":
    unittest.main()
