"""Unit tests for SAM3D pipeline manager precision overrides."""

import os
import unittest

from unittest.mock import patch

from scenesmith.agent_utils.geometry_generation_server.sam3d_pipeline_manager import (
    _get_sam3d_precision_overrides,
)


class TestSAM3DPrecisionOverrides(unittest.TestCase):
    """Verify optional precision overrides and env handling."""

    def test_defaults_to_upstream_pipeline_config(self) -> None:
        """Without env overrides, SceneSmith should preserve upstream defaults."""
        with patch.dict(os.environ, {}, clear=False):
            dtype, shape_dtype = _get_sam3d_precision_overrides()

        self.assertIsNone(dtype)
        self.assertIsNone(shape_dtype)

    def test_shape_dtype_defaults_to_dtype_override(self) -> None:
        """Shape model dtype should inherit the main dtype when unspecified."""
        with patch.dict(os.environ, {"SCENESMITH_SAM3D_DTYPE": "float16"}, clear=False):
            dtype, shape_dtype = _get_sam3d_precision_overrides()

        self.assertEqual(dtype, "float16")
        self.assertEqual(shape_dtype, "float16")

    def test_shape_dtype_can_be_overridden_independently(self) -> None:
        """Advanced users can split global and shape-model precision."""
        with patch.dict(
            os.environ,
            {
                "SCENESMITH_SAM3D_DTYPE": "float16",
                "SCENESMITH_SAM3D_SHAPE_DTYPE": "float32",
            },
            clear=False,
        ):
            dtype, shape_dtype = _get_sam3d_precision_overrides()

        self.assertEqual(dtype, "float16")
        self.assertEqual(shape_dtype, "float32")


if __name__ == "__main__":
    unittest.main()
