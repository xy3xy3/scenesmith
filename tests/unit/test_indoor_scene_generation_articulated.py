"""Unit tests for articulated server config selection."""

import shutil
import tempfile
import unittest

from pathlib import Path

from omegaconf import OmegaConf

from scenesmith.experiments.indoor_scene_generation import (
    _select_articulated_server_config,
)


class TestSelectArticulatedServerConfig(unittest.TestCase):
    """Test articulated server config selection across agent configs."""

    def setUp(self) -> None:
        """Create temporary articulated source directories."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.data_dir = self.temp_dir / "data"
        self.embeddings_dir = self.temp_dir / "embeddings"
        self.data_dir.mkdir(parents=True)
        self.embeddings_dir.mkdir(parents=True)

    def tearDown(self) -> None:
        """Clean up temporary directories."""
        shutil.rmtree(self.temp_dir)

    def _make_agent_cfg(
        self,
        *,
        strategy_enabled: bool,
        source_enabled: bool,
        data_path: Path | None = None,
        embeddings_path: Path | None = None,
    ):
        return {
            "asset_manager": {
                "router": {
                    "strategies": {
                        "articulated": {
                            "enabled": strategy_enabled,
                        }
                    }
                },
                "articulated": {
                    "use_top_k": 5,
                    "sources": {
                        "partnet_mobility": {
                            "enabled": False,
                            "data_path": str(self.data_dir),
                            "embeddings_path": str(self.embeddings_dir),
                        },
                        "artvip": {
                            "enabled": source_enabled,
                            "data_path": str(data_path or self.data_dir),
                            "embeddings_path": str(
                                embeddings_path or self.embeddings_dir
                            ),
                        },
                    },
                },
            }
        }

    def test_returns_none_when_no_usable_sources(self) -> None:
        """All articulated sources disabled should skip server startup."""
        cfg = OmegaConf.create(
            {
                "furniture_agent": self._make_agent_cfg(
                    strategy_enabled=True, source_enabled=False
                ),
                "manipuland_agent": self._make_agent_cfg(
                    strategy_enabled=False, source_enabled=True
                ),
                "wall_agent": self._make_agent_cfg(
                    strategy_enabled=False, source_enabled=True
                ),
                "ceiling_agent": self._make_agent_cfg(
                    strategy_enabled=False, source_enabled=False
                ),
            }
        )

        self.assertIsNone(_select_articulated_server_config(cfg))

    def test_selects_first_agent_with_enabled_strategy_and_usable_sources(self) -> None:
        """Should skip unusable furniture config and fall back to manipuland."""
        missing_dir = self.temp_dir / "missing"
        cfg = OmegaConf.create(
            {
                "furniture_agent": self._make_agent_cfg(
                    strategy_enabled=True,
                    source_enabled=True,
                    data_path=missing_dir,
                    embeddings_path=missing_dir,
                ),
                "manipuland_agent": self._make_agent_cfg(
                    strategy_enabled=True, source_enabled=True
                ),
                "wall_agent": self._make_agent_cfg(
                    strategy_enabled=True, source_enabled=True
                ),
                "ceiling_agent": self._make_agent_cfg(
                    strategy_enabled=False, source_enabled=False
                ),
            }
        )

        selected = _select_articulated_server_config(cfg)
        self.assertIsNotNone(selected)
        agent_name, articulated_cfg = selected
        self.assertEqual(agent_name, "manipuland")
        self.assertTrue(articulated_cfg.sources.artvip.enabled)


if __name__ == "__main__":
    unittest.main()
