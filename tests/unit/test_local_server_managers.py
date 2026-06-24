"""Unit tests for local server manager shutdown and readiness behavior."""

import tempfile
import unittest

from pathlib import Path
from unittest.mock import Mock, patch

from scenesmith.agent_utils.articulated_retrieval_server.server_manager import (
    ArticulatedRetrievalServer,
)
from scenesmith.agent_utils.convex_decomposition_server.server_manager import (
    ConvexDecompositionServer,
)
from scenesmith.agent_utils.geometry_generation_server.server_manager import (
    GeometryGenerationServer,
)
from scenesmith.agent_utils.materials_retrieval_server.server_manager import (
    MaterialsRetrievalServer,
)


class TestLocalServerManagers(unittest.TestCase):
    """Verify server managers stop local WSGI servers directly."""

    def _assert_direct_shutdown(self, server) -> None:
        app = Mock()
        wsgi_server = Mock()
        server_thread = Mock()
        server_thread.is_alive.side_effect = [True, False]

        server._running = True
        server._app = app
        server._server = wsgi_server
        server._server_thread = server_thread

        server.stop()

        app.stop_processing.assert_called_once()
        wsgi_server.shutdown.assert_called_once()
        server_thread.join.assert_called_once_with(timeout=2)
        wsgi_server.server_close.assert_called_once()
        self.assertFalse(server._running)
        self.assertIsNone(server._app)
        self.assertIsNone(server._server)
        self.assertIsNone(server._server_thread)

    def test_materials_server_uses_direct_shutdown(self) -> None:
        """Materials server should stop without relying on HTTP shutdown."""
        self._assert_direct_shutdown(MaterialsRetrievalServer(port=0))

    def test_articulated_server_uses_direct_shutdown(self) -> None:
        """Articulated server should stop without relying on HTTP shutdown."""
        self._assert_direct_shutdown(ArticulatedRetrievalServer(port=0))

    def test_geometry_server_uses_direct_shutdown(self) -> None:
        """Geometry server should stop without relying on HTTP shutdown."""
        self._assert_direct_shutdown(GeometryGenerationServer(port=0))

    @patch(
        "scenesmith.agent_utils.geometry_generation_server.server_manager.requests.get"
    )
    def test_geometry_server_wait_until_ready_uses_requests(self, mock_get) -> None:
        """Geometry server readiness probing should use requests cleanly."""
        response = Mock(status_code=200)
        mock_get.return_value = response

        server = GeometryGenerationServer(port=0)
        server._wait_until_ready(timeout=0.1)

        mock_get.assert_called_once()

    @patch(
        "scenesmith.agent_utils.articulated_retrieval_server.server_manager.requests.get"
    )
    def test_articulated_server_wait_until_ready_uses_requests(self, mock_get) -> None:
        """Articulated server readiness probing should use requests cleanly."""
        response = Mock(status_code=200)
        mock_get.return_value = response

        server = ArticulatedRetrievalServer(port=0)
        server._wait_until_ready(timeout=0.1)

        mock_get.assert_called_once()

    @patch(
        "scenesmith.agent_utils.materials_retrieval_server.server_manager.requests.get"
    )
    def test_materials_server_wait_until_ready_uses_requests(self, mock_get) -> None:
        """Materials server readiness probing should use requests cleanly."""
        response = Mock(status_code=200)
        mock_get.return_value = response

        server = MaterialsRetrievalServer(port=0)
        server._wait_until_ready(timeout=0.1)

        mock_get.assert_called_once()

    @patch(
        "scenesmith.agent_utils.convex_decomposition_server.server_manager.requests.get"
    )
    def test_convex_server_wait_until_ready_uses_requests(self, mock_get) -> None:
        """Convex server readiness probing should use requests cleanly."""
        response = Mock(status_code=200)
        mock_get.return_value = response

        server = ConvexDecompositionServer(port=7100)
        server._running = True
        server._actual_port = 7100
        server._server_process = Mock()
        server._server_process.poll.return_value = None

        server.wait_until_ready(timeout=0.1, poll_interval=0.01)

        mock_get.assert_called_once()

    def test_convex_server_wait_until_ready_reports_early_process_exit(self) -> None:
        """Convex server readiness failure should include exit status and log tail."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "convex.log"
            log_file.write_text("startup failed\ntraceback line\n")

            server = ConvexDecompositionServer(port=7100, log_file=log_file)
            server._running = True
            server._actual_port = 7100
            server._server_process = Mock()
            server._server_process.poll.return_value = 1

            with self.assertRaises(RuntimeError) as context:
                server.wait_until_ready(timeout=1.0, poll_interval=0.01)

            message = str(context.exception)
            self.assertIn("exited before becoming ready", message)
            self.assertIn("Exited with code 1", message)
            self.assertIn("startup failed", message)


if __name__ == "__main__":
    unittest.main()
