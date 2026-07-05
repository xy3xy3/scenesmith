"""Unit tests for the asset router module."""

import tempfile
import unittest

from pathlib import Path
from unittest.mock import MagicMock

from omegaconf import OmegaConf

from scenesmith.agent_utils.asset_router import AssetRouter
from scenesmith.agent_utils.asset_router.dataclasses import AnalysisResult, AssetItem
from scenesmith.agent_utils.room import AgentType, ObjectType
from scenesmith.agent_utils.simple_manipuland_primitives import (
    can_generate_simple_manipuland_primitive,
    generate_simple_manipuland_primitive,
)


class TestAnalysisResultWasModified(unittest.TestCase):
    """Test the was_modified computed property logic."""

    def test_single_item_not_modified(self) -> None:
        """Single item with no original_description is not modified."""
        item = AssetItem(
            description="wooden ladder",
            short_name="ladder",
            dimensions=[0.5, 0.3, 2.0],
            object_type=ObjectType.FURNITURE,
            strategies=["generated"],
        )
        result = AnalysisResult(
            items=[item],
            original_description=None,
            discarded_manipulands=None,
        )
        assert not result.was_modified

    def test_with_original_description_is_modified(self) -> None:
        """Items with original_description set is modified (was split/filtered)."""
        items = [
            AssetItem(
                description="dining table",
                short_name="dining_table",
                dimensions=[1.5, 0.9, 0.75],
                object_type=ObjectType.FURNITURE,
                strategies=["generated"],
            ),
        ]
        result = AnalysisResult(
            items=items,
            original_description="dining table and four chairs",
            discarded_manipulands=None,
        )
        assert result.was_modified

    def test_with_discarded_manipulands_is_modified(self) -> None:
        """Request with discarded manipulands is modified."""
        item = AssetItem(
            description="ladder",
            short_name="ladder",
            dimensions=[0.5, 0.3, 2.0],
            object_type=ObjectType.FURNITURE,
            strategies=["generated"],
        )
        result = AnalysisResult(
            items=[item],
            original_description="ladder with flower pots",
            discarded_manipulands=["flower pots"],
        )
        assert result.was_modified


class TestAssetRouterItemTypeValidation(unittest.TestCase):
    """Test validate_item_types method behavior."""

    def test_furniture_items_valid_for_furniture_agent(self) -> None:
        """Furniture items are valid for furniture agent."""
        router = AssetRouter(
            agent_type=AgentType.FURNITURE, vlm_service=MagicMock(), cfg=MagicMock()
        )

        items = [
            AssetItem(
                description="desk",
                short_name="desk",
                dimensions=[1.2, 0.6, 0.75],
                object_type=ObjectType.FURNITURE,
                strategies=["generated"],
            ),
        ]

        error = router.validate_item_types(items)
        assert error is None

    def test_manipuland_items_valid_for_manipuland_agent(self) -> None:
        """Manipuland items are valid for manipuland agent."""
        router = AssetRouter(
            agent_type=AgentType.MANIPULAND, vlm_service=MagicMock(), cfg=MagicMock()
        )

        items = [
            AssetItem(
                description="coffee mug",
                short_name="mug",
                dimensions=[0.08, 0.08, 0.1],
                object_type=ObjectType.MANIPULAND,
                strategies=["generated"],
            ),
        ]

        error = router.validate_item_types(items)
        assert error is None

    def test_either_type_valid_for_both_agents(self) -> None:
        """EITHER type items are valid for both furniture and manipuland agents."""
        item = AssetItem(
            description="potted plant",
            short_name="potted_plant",
            dimensions=[0.3, 0.3, 0.6],
            object_type=ObjectType.EITHER,
            strategies=["generated"],
        )

        furniture_router = AssetRouter(
            agent_type=AgentType.FURNITURE, vlm_service=MagicMock(), cfg=MagicMock()
        )
        assert furniture_router.validate_item_types([item]) is None

        manipuland_router = AssetRouter(
            agent_type=AgentType.MANIPULAND, vlm_service=MagicMock(), cfg=MagicMock()
        )
        assert manipuland_router.validate_item_types([item]) is None

    def test_wrong_type_returns_error(self) -> None:
        """Wrong item type for agent returns error message."""
        router = AssetRouter(
            agent_type=AgentType.FURNITURE, vlm_service=MagicMock(), cfg=MagicMock()
        )

        items = [
            AssetItem(
                description="coffee mug",
                short_name="mug",
                dimensions=[0.08, 0.08, 0.1],
                object_type=ObjectType.MANIPULAND,
                strategies=["generated"],
            ),
        ]

        error = router.validate_item_types(items)
        assert error is not None
        assert "manipuland" in error.lower()


class TestSimpleManipulandPrimitiveFallback(unittest.TestCase):
    """Test deterministic primitive fallback for simple manipulands."""

    def _make_router(self) -> AssetRouter:
        cfg = OmegaConf.create(
            {
                "asset_manager": {
                    "general_asset_source": "hssd",
                    "side_view_elevation_degrees": 15.0,
                    "validation_taa_samples": 8,
                    "router": {
                        "simple_manipuland_fallback": {"enabled": True},
                        "strategies": {
                            "generated": {"enabled": True, "max_retries": 1}
                        },
                    },
                    "hssd": {"use_lenient_validation": True},
                    "objaverse": {"use_lenient_validation": True},
                }
            }
        )
        return AssetRouter(
            agent_type=AgentType.MANIPULAND,
            vlm_service=MagicMock(),
            cfg=cfg,
        )

    def test_primitive_classifier_is_conservative(self) -> None:
        coaster = AssetItem(
            description="round drink coasters",
            short_name="coasters",
            dimensions=[0.09, 0.09, 0.01],
            object_type=ObjectType.MANIPULAND,
            strategies=["generated"],
        )
        sculpture = AssetItem(
            description="abstract decorative sculpture",
            short_name="decor_sculpture",
            dimensions=[0.2, 0.2, 0.4],
            object_type=ObjectType.MANIPULAND,
            strategies=["generated"],
        )

        self.assertTrue(can_generate_simple_manipuland_primitive(coaster))
        self.assertFalse(can_generate_simple_manipuland_primitive(sculpture))

    def test_generate_simple_primitive_glb(self) -> None:
        item = AssetItem(
            description="television remote control",
            short_name="remote_control",
            dimensions=[0.18, 0.05, 0.015],
            object_type=ObjectType.MANIPULAND,
            strategies=["generated"],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = generate_simple_manipuland_primitive(item, Path(tmpdir))

            self.assertIsNotNone(path)
            self.assertTrue(path.exists())
            self.assertEqual(path.suffix, ".glb")

    def test_router_uses_primitive_fallback_when_retrieval_fails(self) -> None:
        router = self._make_router()
        item = AssetItem(
            description="round drink coasters",
            short_name="coasters",
            dimensions=[0.09, 0.09, 0.01],
            object_type=ObjectType.MANIPULAND,
            strategies=["generated"],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            result = router.generate_with_validation(
                item=item,
                geometry_client=None,
                image_generator=None,
                images_dir=None,
                geometry_dir=Path(tmpdir),
                debug_dir=Path(tmpdir) / "debug",
                hssd_client=None,
            )

            self.assertIsNotNone(result)
            self.assertEqual(result.asset_source, "procedural_primitive")
            self.assertTrue(result.geometry_path.exists())


class TestAnalysisResponseParsing(unittest.TestCase):
    """Test parsing of VLM analysis responses."""

    def _make_router(self, agent_type: AgentType) -> AssetRouter:
        cfg = OmegaConf.create(
            {
                "asset_manager": {
                    "side_view_elevation_degrees": 15.0,
                    "validation_taa_samples": 8,
                    "manipuland_scale": {
                        "enabled": True,
                        "normalize_router_dimensions": True,
                        "reject_bad_uniform_scale_fit": True,
                        "max_axis_relative_error": 0.75,
                    },
                }
            }
        )
        return AssetRouter(agent_type=agent_type, vlm_service=MagicMock(), cfg=cfg)

    def test_parse_single_furniture_item(self) -> None:
        """Parse single furniture item response."""
        router = AssetRouter(
            agent_type=AgentType.FURNITURE, vlm_service=MagicMock(), cfg=MagicMock()
        )

        response = {
            "items": [
                {
                    "description": "wooden ladder",
                    "short_name": "ladder",
                    "dimensions": [0.5, 0.3, 2.0],
                    "object_type": "FURNITURE",
                    "strategies": ["generated"],
                }
            ],
            "original_description": None,
            "discarded_manipulands": None,
        }

        result = router._parse_analysis_response(response)
        assert len(result.items) == 1
        assert result.items[0].description == "wooden ladder"
        assert result.items[0].object_type == ObjectType.FURNITURE
        assert not result.was_modified

    def test_parse_composite_split(self) -> None:
        """Parse response with composite split into multiple items."""
        router = AssetRouter(
            agent_type=AgentType.MANIPULAND, vlm_service=MagicMock(), cfg=MagicMock()
        )

        response = {
            "items": [
                {
                    "description": "fruit bowl",
                    "short_name": "fruit_bowl",
                    "dimensions": [0.3, 0.3, 0.10],
                    "object_type": "MANIPULAND",
                    "strategies": ["generated"],
                },
                {
                    "description": "apple",
                    "short_name": "apple",
                    "dimensions": [0.08, 0.08, 0.08],
                    "object_type": "MANIPULAND",
                    "strategies": ["generated"],
                },
            ],
            "original_description": "fruit bowl with apples",
        }

        result = router._parse_analysis_response(response)
        assert len(result.items) == 2
        assert result.was_modified
        assert result.original_description == "fruit bowl with apples"

    def test_parse_error_response(self) -> None:
        """Parse error response from VLM."""
        router = AssetRouter(
            agent_type=AgentType.FURNITURE, vlm_service=MagicMock(), cfg=MagicMock()
        )

        response = {
            "items": [],
            "original_description": None,
            "discarded_manipulands": None,
            "error": "Request is for a manipuland (coffee mug), not furniture.",
        }

        result = router._parse_analysis_response(response)
        assert len(result.items) == 0
        assert result.error is not None
        assert "manipuland" in result.error.lower()

    def test_parse_error_response_preserves_original_description(self) -> None:
        """Error responses preserve original_description for debugging."""
        router = AssetRouter(
            agent_type=AgentType.FURNITURE, vlm_service=MagicMock(), cfg=MagicMock()
        )

        response = {
            "items": [],
            "original_description": "stack of 4 car tires",
            "discarded_manipulands": None,
            "error": "Stackable items should be handled by manipuland agent.",
        }

        result = router._parse_analysis_response(response)
        assert len(result.items) == 0
        assert result.error is not None
        assert result.original_description == "stack of 4 car tires"

    def test_parse_with_discarded_manipulands(self) -> None:
        """Parse response with discarded manipulands (furniture agent filtering)."""
        router = AssetRouter(
            agent_type=AgentType.FURNITURE, vlm_service=MagicMock(), cfg=MagicMock()
        )

        response = {
            "items": [
                {
                    "description": "bookshelf",
                    "short_name": "bookshelf",
                    "dimensions": [1.0, 0.3, 2.0],
                    "object_type": "FURNITURE",
                    "strategies": ["generated"],
                }
            ],
            "original_description": "bookshelf with books and decorations",
            "discarded_manipulands": ["books", "decorations"],
        }

        result = router._parse_analysis_response(response)
        assert len(result.items) == 1
        assert result.was_modified
        assert result.discarded_manipulands == ["books", "decorations"]

    def test_parse_lowercase_object_type(self) -> None:
        """Object type parsing is case-insensitive."""
        router = AssetRouter(
            agent_type=AgentType.FURNITURE, vlm_service=MagicMock(), cfg=MagicMock()
        )

        response = {
            "items": [
                {
                    "description": "desk",
                    "short_name": "desk",
                    "dimensions": [1.2, 0.6, 0.75],
                    "object_type": "furniture",  # lowercase
                    "strategies": ["generated"],
                }
            ],
            "original_description": None,
        }

        result = router._parse_analysis_response(response)
        assert len(result.items) == 1
        assert result.items[0].object_type == ObjectType.FURNITURE

    def test_parse_manipuland_normalizes_known_scale_profile(self) -> None:
        """Known manipulands keep original request and use clamped dimensions."""
        router = self._make_router(AgentType.MANIPULAND)

        response = {
            "items": [
                {
                    "description": "spiral notebook",
                    "short_name": "notebook",
                    "dimensions": [0.60, 0.04, 0.20],
                    "object_type": "MANIPULAND",
                    "strategies": ["generated"],
                }
            ],
            "original_description": None,
        }

        result = router._parse_analysis_response(response)

        assert len(result.items) == 1
        item = result.items[0]
        assert item.requested_dimensions == [0.60, 0.04, 0.20]
        assert item.dimensions == [0.30, 0.09, 0.06]
        assert item.scale_profile == "notebook_book"


class TestAssetRouterJsonFallback(unittest.TestCase):
    """Test JSON fallback behavior for local/open model outputs."""

    def _make_router(self) -> AssetRouter:
        cfg = OmegaConf.create(
            {
                "openai": {
                    "model": "test-model",
                    "reasoning_effort": {
                        "asset_analysis": "low",
                        "asset_validation": "low",
                    },
                    "verbosity": {
                        "asset_analysis": "low",
                        "asset_validation": "low",
                    },
                    "vision_detail": "low",
                },
                "asset_manager": {
                    "side_view_elevation_degrees": 15.0,
                    "validation_taa_samples": 8,
                    "router": {"analysis_max_retries": 1},
                },
            }
        )
        return AssetRouter(
            agent_type=AgentType.FURNITURE,
            vlm_service=MagicMock(),
            cfg=cfg,
        )

    def test_analyze_description_accepts_fenced_json(self) -> None:
        """Router analysis should accept fenced JSON from local models."""
        router = self._make_router()
        router.vlm_service.create_completion.return_value = """```json
        {
          "items": [
            {
              "description": "desk",
              "short_name": "desk",
              "dimensions": [1.2, 0.6, 0.75],
              "object_type": "FURNITURE",
              "strategies": ["generated"]
            }
          ],
          "original_description": null,
          "discarded_manipulands": null
        }
        ```"""

        result = router.analyze_request("desk", [1.2, 0.6, 0.75])

        assert len(result.items) == 1
        assert result.items[0].description == "desk"


if __name__ == "__main__":
    unittest.main()
