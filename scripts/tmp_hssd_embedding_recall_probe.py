#!/usr/bin/env python3
"""Temporary probe for HSSD embedding recall on a few manual queries.

Serially queries the local HSSD Zvec collection using the llama.cpp embedding
server, saves per-query JSON results, and creates simple montage images from
the rendered asset views for quick human inspection.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import time
import urllib.error
import urllib.request

from pathlib import Path
from typing import Any

import zvec

from PIL import Image, ImageDraw

QUERY_MAP = {
    "电视": "television",
    "叉子": "fork",
    "刀": "knife",
    "筷子": "chopsticks",
    "碗": "bowl",
    "笔记本电脑": "laptop",
    "显示器": "computer monitor",
}


def _extract_embedding(response: Any) -> list[float]:
    if isinstance(response, list) and response and isinstance(response[0], dict):
        embedding = response[0].get("embedding") or response[0].get("embeddings")
    elif isinstance(response, list):
        embedding = response
    elif isinstance(response, dict):
        embedding = (
            response.get("embedding")
            or response.get("embeddings")
            or (response.get("data") or [{}])[0].get("embedding")
        )
    else:
        embedding = None

    while (
        isinstance(embedding, list)
        and len(embedding) == 1
        and isinstance(embedding[0], list)
    ):
        embedding = embedding[0]

    if not isinstance(embedding, list):
        return []
    return [float(value) for value in embedding]


class LlamaTextEmbeddingClient:
    def __init__(
        self,
        base_url: str,
        timeout_seconds: float = 120.0,
        embd_normalize: int = 2,
        request_retries: int = 2,
        retry_sleep_seconds: float = 1.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.embd_normalize = embd_normalize
        self.request_retries = request_retries
        self.retry_sleep_seconds = retry_sleep_seconds

    def embed_text(self, text: str) -> list[float]:
        payloads = [
            {"content": text, "embd_normalize": self.embd_normalize},
            {
                "content": {"prompt_string": text, "multimodal_data": []},
                "embd_normalize": self.embd_normalize,
            },
            {"input": text, "embd_normalize": self.embd_normalize},
        ]

        last_error: Exception | None = None
        for payload in payloads:
            try:
                body = self._post_embeddings(payload)
                embedding = _extract_embedding(json.loads(body))
                if embedding:
                    return embedding
                last_error = RuntimeError("Empty embedding response")
            except Exception as exc:
                last_error = exc

        if last_error is not None:
            raise last_error
        raise RuntimeError("Failed to embed text query")

    def _post_embeddings(self, payload: dict[str, Any]) -> str:
        request = urllib.request.Request(
            f"{self.base_url}/embeddings",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        last_error: Exception | None = None
        for attempt in range(self.request_retries + 1):
            try:
                with urllib.request.urlopen(
                    request, timeout=self.timeout_seconds
                ) as response:
                    return response.read().decode("utf-8")
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                last_error = RuntimeError(
                    f"Embedding request failed: HTTP {exc.code}: {detail[:500]}"
                )
                if 400 <= exc.code < 500 and exc.code not in {408, 429}:
                    raise last_error from exc
            except (TimeoutError, urllib.error.URLError) as exc:
                last_error = RuntimeError(f"Embedding request failed: {exc}")

            if attempt < self.request_retries:
                time.sleep(self.retry_sleep_seconds * (attempt + 1))

        if last_error is not None:
            raise last_error
        raise RuntimeError("Embedding request failed")


def make_montage(items: list[dict[str, Any]], output_path: Path, thumb_size: int = 256):
    views = ["front", "top"]
    cols = len(views)
    rows = len(items)
    pad = 16
    label_h = 70
    width = cols * thumb_size + (cols + 1) * pad
    height = rows * (thumb_size + label_h) + (rows + 1) * pad
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)

    for row, item in enumerate(items):
        y0 = pad + row * (thumb_size + label_h + pad)
        label = f"{row+1}. {item['asset_id'][:8]}  score={item['score']:.4f}\n{item['name'] or '(no name)'}"
        draw.text((pad, y0), label, fill="black")
        for col, view in enumerate(views):
            x = pad + col * (thumb_size + pad)
            img_y = y0 + label_h
            img_path = item["view_paths"].get(view)
            if img_path and Path(img_path).exists():
                img = Image.open(img_path).convert("RGB")
                img.thumbnail((thumb_size, thumb_size))
                tile = Image.new("RGB", (thumb_size, thumb_size), "white")
                ox = (thumb_size - img.width) // 2
                oy = (thumb_size - img.height) // 2
                tile.paste(img, (ox, oy))
                canvas.paste(tile, (x, img_y))
                draw.rectangle(
                    [(x, img_y), (x + thumb_size - 1, img_y + thumb_size - 1)],
                    outline="gray",
                    width=1,
                )
                draw.text((x + 6, img_y + 6), view, fill="black")
            else:
                draw.rectangle(
                    [(x, img_y), (x + thumb_size - 1, img_y + thumb_size - 1)],
                    outline="red",
                    width=2,
                )
                draw.text((x + 8, img_y + 8), f"missing {view}", fill="red")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def write_html_report(all_results: list[dict[str, Any]], output_path: Path) -> None:
    lines = [
        "<html><head><meta charset='utf-8'><title>HSSD Recall Probe</title></head><body>",
        "<h1>HSSD Recall Probe</h1>",
    ]
    for result in all_results:
        lines.append(
            f"<h2>{html.escape(result['query_zh'])} / {html.escape(result['query_en'])}</h2>"
        )
        montage_rel = output_path.parent.relative_to(output_path.parent) / Path(
            result["montage_path"]
        ).name
        lines.append(f"<p><img src='{html.escape(montage_rel.as_posix())}' width='900'></p>")
        lines.append("<ol>")
        for item in result["results"]:
            lines.append(
                "<li>"
                f"score={item['score']:.4f} "
                f"asset_id={html.escape(item['asset_id'])} "
                f"name={html.escape(item['name'] or '')} "
                f"wordnet={html.escape(item['wordnet_key'] or '')}"
                "</li>"
            )
        lines.append("</ol>")
    lines.append("</body></html>")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Temporary HSSD embedding recall probe")
    parser.add_argument(
        "--collection-path",
        type=Path,
        default=Path("data/hssd_zvec_collection"),
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8014",
    )
    parser.add_argument(
        "--render-root",
        type=Path,
        default=Path("data/hssd_rendered_assets"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/hssd_recall_probe"),
    )
    parser.add_argument("--topk", type=int, default=8)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    client = LlamaTextEmbeddingClient(base_url=args.base_url)
    collection = zvec.open(
        path=str(args.collection_path),
        option=zvec.CollectionOption(read_only=True, enable_mmap=True),
    )

    all_results: list[dict[str, Any]] = []

    for query_zh, query_en in QUERY_MAP.items():
        print(f"Querying: {query_zh} -> {query_en}")
        query_embedding = client.embed_text(query_en)
        docs = collection.query(
            queries=zvec.Query(field_name="embedding", vector=query_embedding),
            topk=args.topk,
            output_fields=[
                "asset_id",
                "name",
                "wordnet_key",
                "object_groups",
                "views",
                "image_paths",
                "asset_path",
            ],
            include_vector=False,
        )

        items: list[dict[str, Any]] = []
        for doc in docs:
            fields = doc.fields or {}
            asset_id = fields.get("asset_id") or doc.id
            image_paths = fields.get("image_paths") or []
            view_paths: dict[str, str] = {}
            for img_path in image_paths:
                path = Path(img_path)
                view_paths[path.stem] = str(path)

            if "front" not in view_paths:
                front = args.render_root / asset_id / "front.png"
                if front.exists():
                    view_paths["front"] = str(front)
            if "top" not in view_paths:
                top = args.render_root / asset_id / "top.png"
                if top.exists():
                    view_paths["top"] = str(top)

            items.append(
                {
                    "rank": len(items) + 1,
                    "asset_id": asset_id,
                    "score": float(doc.score or 0.0),
                    "name": fields.get("name", ""),
                    "wordnet_key": fields.get("wordnet_key", ""),
                    "object_groups": fields.get("object_groups", []),
                    "asset_path": fields.get("asset_path", ""),
                    "view_paths": view_paths,
                }
            )

        json_path = args.output_dir / f"{query_en.replace(' ', '_')}.json"
        montage_path = args.output_dir / f"{query_en.replace(' ', '_')}.png"
        json_path.write_text(
            json.dumps(
                {
                    "query_zh": query_zh,
                    "query_en": query_en,
                    "results": items,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        make_montage(items, montage_path)
        all_results.append(
            {
                "query_zh": query_zh,
                "query_en": query_en,
                "results": items,
                "montage_path": str(montage_path),
            }
        )

    write_html_report(all_results, args.output_dir / "report.html")
    print(f"Wrote report to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
