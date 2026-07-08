from argparse import Namespace
from unittest.mock import MagicMock, patch

import numpy as np

from scenesmith.agent_utils.hssd_retrieval.data_loader import HssdPreprocessedData
from scripts import audit_hssd_critic_fronts


def _make_preprocessed_data() -> HssdPreprocessedData:
    return HssdPreprocessedData(
        metadata_by_wordnet={},
        clip_embeddings=np.zeros((0, 0), dtype=np.float32),
        embedding_index=[],
        object_categories={"large_objects": ["bed.n.01"]},
    )


def test_effective_name_filter_defaults_to_disabled() -> None:
    assert audit_hssd_critic_fronts._effective_name_filter("bed", None, False) == []


def test_effective_retrieval_query_uses_stable_phrase_for_bed() -> None:
    assert audit_hssd_critic_fronts._effective_retrieval_query("bed") == "double bed"


def test_resolve_asset_list_for_keyword_uses_semantic_only_by_default() -> None:
    args = Namespace(
        name_filter=None,
        no_default_filter=False,
        search_mode="zvec",
        top_k=5,
        sample_count=2,
        seed=7,
    )
    with patch(
        "scripts.audit_hssd_critic_fronts.retrieve_assets_zvec",
        return_value=[("bed_mesh", 0.91)],
    ) as retrieve_assets_zvec:
        result = audit_hssd_critic_fronts.resolve_asset_list_for_keyword(
            query="bed",
            args=args,
            metadata_by_id={},
            preprocessed_data=_make_preprocessed_data(),
            zvec_searcher=MagicMock(),
        )

        assert result == [("bed_mesh", 0.91)]
        retrieve_assets_zvec.assert_called_once()
        assert retrieve_assets_zvec.call_args.kwargs["query"] == "double bed"
        assert retrieve_assets_zvec.call_args.kwargs["name_filter"] == []
        assert retrieve_assets_zvec.call_args.kwargs["hssd_category"] == "large_objects"


def test_resolve_asset_list_for_keyword_keeps_explicit_name_filter() -> None:
    args = Namespace(
        name_filter="bed,mattress",
        no_default_filter=False,
        search_mode="clip",
        top_k=5,
        sample_count=2,
        seed=7,
    )
    with patch(
        "scripts.audit_hssd_critic_fronts.retrieve_assets_clip",
        return_value=[("bed_mesh", 0.88)],
    ) as retrieve_assets_clip:
        result = audit_hssd_critic_fronts.resolve_asset_list_for_keyword(
            query="bed",
            args=args,
            metadata_by_id={},
            preprocessed_data=_make_preprocessed_data(),
            zvec_searcher=None,
        )

        assert result == [("bed_mesh", 0.88)]
        retrieve_assets_clip.assert_called_once()
        assert retrieve_assets_clip.call_args.kwargs["query"] == "double bed"
        assert retrieve_assets_clip.call_args.kwargs["name_filter"] == [
            "bed",
            "mattress",
        ]
