from __future__ import annotations

import argparse
import json
from pathlib import Path

import onnxmltools
import xgboost as xgb
from onnxmltools.convert.common.data_types import FloatTensorType


def export_xgboost_to_onnx(
    model_path: Path,
    output_path: Path,
    embedding_dim: int = 768,
) -> None:
    if not model_path.exists():
        raise FileNotFoundError(f"XGBoost model not found: {model_path}")

    classifier = xgb.XGBClassifier()
    classifier.load_model(str(model_path))

    initial_types = [("embedding", FloatTensorType([None, embedding_dim]))]
    onnx_model = onnxmltools.convert_xgboost(
        classifier,
        initial_types=initial_types,
        target_opset=15,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(onnx_model.SerializeToString())

    metadata = {
        "source_model": model_path.as_posix(),
        "onnx_model": output_path.as_posix(),
        "input_name": "embedding",
        "input_shape": [None, embedding_dim],
        "input_description": "DeBERTa-v3 pooled embedding generated from model_text",
        "label_mapping": {"non-bug": 0, "bug": 1},
    }
    metadata_path = output_path.with_suffix(".metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"ONNX model written to: {output_path}")
    print(f"ONNX metadata written to: {metadata_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export the trained XGBoost baseline classifier to ONNX."
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("/kaggle/working/outputs/models/baseline/full/xgb_baseline.json"),
        help="Path to trained XGBoost JSON model.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/kaggle/working/outputs/models/baseline/full/xgb_baseline.onnx"),
        help="Output ONNX model path.",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=768,
        help="Embedding dimension. DeBERTa-v3-small uses 768.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    export_xgboost_to_onnx(args.model, args.output, args.embedding_dim)
