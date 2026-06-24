import unittest

from types import SimpleNamespace
from unittest.mock import Mock
from unittest.mock import patch

import numpy as np

with patch.dict("sys.modules", {"coacd": SimpleNamespace(set_log_level=lambda *_: None)}):
    from scenesmith.agent_utils.convex_decomposition_server.server_app import (
        ConvexDecompositionServerApp,
    )


class TestConvexDecompositionServerApp(unittest.TestCase):
    """Focused regression tests for convex decomposition server app."""

    def test_run_vhacd_serializes_numpy_arrays(self) -> None:
        """V-HACD results should serialize even with lazy numpy imports."""
        app = ConvexDecompositionServerApp()
        mesh = Mock()

        piece_mesh = Mock()
        piece_mesh.vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        piece_mesh.faces = np.array([[0, 1, 1]])
        mesh.convex_decomposition.return_value = [piece_mesh]

        pieces = app._run_vhacd(mesh, {})

        mesh.convex_decomposition.assert_called_once()
        self.assertEqual(
            pieces,
            [
                {
                    "vertices": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                    "faces": [[0, 1, 1]],
                }
            ],
        )


if __name__ == "__main__":
    unittest.main()
