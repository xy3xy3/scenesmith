import asyncio
import shutil
import tempfile
import unittest

from pathlib import Path
from unittest.mock import ANY, MagicMock, patch

from omegaconf import OmegaConf

from scenesmith.agent_utils.room import AgentType, RoomScene
from scenesmith.manipuland_agents.base_manipuland_agent import BaseManipulandAgent
from tests.unit.mock_utils import create_mock_logger


class ConcreteManipulandAgent(BaseManipulandAgent):
    """Concrete implementation for testing abstract base class."""

    async def add_manipulands(self, scene):
        """Test implementation."""
        return "Test manipulands added"


class TestBaseManipulandAgent(unittest.TestCase):
    """Test BaseManipulandAgent class."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.mock_logger = create_mock_logger(self.temp_dir)

        # Load configuration from actual config file.
        config_path = (
            Path(__file__).parent.parent.parent
            / "configurations/manipuland_agent/base_manipuland_agent.yaml"
        )
        base_config = OmegaConf.load(config_path)

        # Note: service_tier in agent configs references ${openai.service_tier} from
        # the top-level config.yaml which isn't loaded in tests. Provide both the
        # top-level key and override the interpolation in the agent config.
        test_overrides = {
            "openai": {
                "service_tier": None,  # Top-level openai.service_tier for interpolation
            },
            "manipuland_agent": {
                "openai": {
                    "service_tier": None,  # Override interpolation directly
                },
            },
        }
        self.config = OmegaConf.merge(base_config, test_overrides)

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch("scenesmith.manipuland_agents.base_manipuland_agent.AssetManager")
    @patch("scenesmith.manipuland_agents.base_manipuland_agent.VLMService")
    @patch("scenesmith.manipuland_agents.base_manipuland_agent.RenderingManager")
    @patch(
        "scenesmith.manipuland_agents.base_manipuland_agent.ConvexDecompositionServer"
    )
    @patch("scenesmith.manipuland_agents.base_manipuland_agent.BlenderServer")
    def test_initialization(
        self,
        mock_blender_server_class,
        mock_convex_decomposition_server_class,
        mock_rendering_manager_class,
        mock_vlm_service_class,
        mock_asset_manager_class,
    ):
        """Test BaseManipulandAgent initialization."""
        # Configure mock BlenderServer.
        mock_blender_server_class.return_value.is_running.return_value = True

        agent = ConcreteManipulandAgent(cfg=self.config, logger=self.mock_logger)

        self.assertEqual(agent.cfg, self.config)
        self.assertEqual(agent.logger, self.mock_logger)

        # Verify dependencies were created.
        mock_vlm_service_class.assert_called_once()
        mock_convex_decomposition_server_class.assert_called_once()
        mock_blender_server_class.assert_called_once_with(
            port_range=tuple(self.config.rendering.blender_server_port_range),
            server_startup_delay=self.config.rendering.server_startup_delay,
            port_cleanup_delay=self.config.rendering.port_cleanup_delay,
            gpu_id=None,
            log_file=self.mock_logger.output_dir / "blender_server.log",
        )
        mock_blender_server_class.return_value.wait_until_ready.assert_called_once_with(
            timeout=180.0,
            poll_interval=1.0,
        )
        mock_asset_manager_class.assert_called_once_with(
            logger=self.mock_logger,
            vlm_service=mock_vlm_service_class.return_value,
            blender_server=ANY,
            collision_client=ANY,
            cfg=self.config,
            agent_type=AgentType.MANIPULAND,
            geometry_server_host="127.0.0.1",
            geometry_server_port=7000,
            hssd_server_host="127.0.0.1",
            hssd_server_port=7001,
            articulated_server_host="127.0.0.1",
            articulated_server_port=7002,
            materials_server_enabled=True,
            materials_server_host="127.0.0.1",
            materials_server_port=7008,
        )
        mock_rendering_manager_class.assert_called_once_with(
            cfg=self.config.rendering, logger=self.mock_logger
        )

    @patch("scenesmith.manipuland_agents.base_manipuland_agent.AssetManager")
    @patch("scenesmith.manipuland_agents.base_manipuland_agent.VLMService")
    @patch("scenesmith.manipuland_agents.base_manipuland_agent.RenderingManager")
    @patch(
        "scenesmith.manipuland_agents.base_manipuland_agent.ConvexDecompositionServer"
    )
    @patch("scenesmith.manipuland_agents.base_manipuland_agent.BlenderServer")
    def test_abstract_method_implemented(
        self,
        mock_blender_server_class,
        mock_convex_decomposition_server_class,
        mock_rendering_manager_class,
        mock_vlm_service_class,
        mock_asset_manager_class,
    ):
        """Test that concrete class implements abstract method."""
        # Configure mock BlenderServer.
        mock_blender_server_class.return_value.is_running.return_value = True

        agent = ConcreteManipulandAgent(cfg=self.config, logger=self.mock_logger)

        # Should be able to call add_manipulands without TypeError.
        mock_scene = MagicMock(spec=RoomScene)
        result = asyncio.run(agent.add_manipulands(mock_scene))
        self.assertIsNotNone(result)

    def test_abstract_method_not_implemented_raises_error(self):
        """Test that instantiating abstract class directly raises TypeError."""
        with self.assertRaises(TypeError):
            BaseManipulandAgent(cfg=self.config, logger=self.mock_logger)

    @patch("scenesmith.manipuland_agents.base_manipuland_agent.AssetManager")
    @patch("scenesmith.manipuland_agents.base_manipuland_agent.VLMService")
    @patch("scenesmith.manipuland_agents.base_manipuland_agent.RenderingManager")
    @patch(
        "scenesmith.manipuland_agents.base_manipuland_agent.ConvexDecompositionServer"
    )
    @patch("scenesmith.manipuland_agents.base_manipuland_agent.BlenderServer")
    @patch("scenesmith.manipuland_agents.base_manipuland_agent.os.cpu_count")
    def test_collision_server_startup_uses_configured_limits(
        self,
        mock_cpu_count,
        mock_blender_server_class,
        mock_convex_decomposition_server_class,
        mock_rendering_manager_class,
        mock_vlm_service_class,
        mock_asset_manager_class,
    ):
        """Auto-selected OMP threads and readiness timeout should come from config."""
        mock_cpu_count.return_value = 192
        mock_blender_server_class.return_value.is_running.return_value = True
        self.config.collision_geometry.startup.max_omp_threads = 8
        self.config.collision_geometry.startup.ready_timeout_s = 45.0
        self.config.collision_geometry.startup.poll_interval_s = 0.25

        agent = ConcreteManipulandAgent(cfg=self.config, logger=self.mock_logger)

        mock_convex_decomposition_server_class.assert_called_once_with(
            port_range=tuple(self.config.collision_geometry.server_port_range),
            omp_threads=8,
            log_file=self.mock_logger.output_dir / "room.log",
        )
        mock_convex_decomposition_server_class.return_value.wait_until_ready.assert_called_once_with(
            timeout=45.0,
            poll_interval=0.25,
        )
        self.assertIsNotNone(agent)


if __name__ == "__main__":
    unittest.main()
