"""Section 4, Text embeddings.

Generates narrative embeddings via a HuggingFace sentence-transformers model
for the fused model (Stage 2). Complaints with no narrative get a zero
vector, per BUILD_INSTRUCTIONS.md Architecture, no-narrative complaints are
never dropped from the dataset, only zero-imputed at this stage.

Model: sentence-transformers/all-MiniLM-L6-v2 (384-dim). Note: this model's
max input length is 256 tokens, longer narratives are silently truncated by
the tokenizer. See README.md Explicit Limitations for the full discussion;
Section 6 (Evaluate) splits results by narrative length to surface whether
this masks the fused model's benefit on long narratives specifically.

Acceptance criteria this module supports (see BUILD_INSTRUCTIONS.md):
- Given a complaint with an empty narrative, the fused-model pipeline does
  not crash and the row is not dropped from the structured comparison.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import load_config

logger = logging.getLogger(__name__)


class EmbeddingsError(Exception):
    """Raised when the labeled dataset is missing, empty, or malformed."""


def load_labeled_data(path: Path) -> list[dict[str, Any]]:
    """Load the labeled dataset, raising EmbeddingsError on missing/empty/malformed input."""
    if not path.exists():
        raise EmbeddingsError(
            f"Labeled data file not found at {path}. Run `python -m src.label` first."
        )

    with open(path) as f:
        try:
            records = json.load(f)
        except json.JSONDecodeError as e:
            raise EmbeddingsError(f"Labeled data file at {path} is not valid JSON: {e}") from e

    if not records:
        raise EmbeddingsError(f"Labeled data file at {path} is empty.")
    if not isinstance(records, list):
        raise EmbeddingsError(
            f"Labeled data file at {path} does not contain a list of records "
            f"(got {type(records).__name__})."
        )
    return records


def generate_embeddings(
    records: list[dict[str, Any]], model_name: str, dimension: int
) -> np.ndarray:
    """Encode each record's narrative into a `dimension`-dim vector.

    Records with no narrative (empty/None complaint_what_happened) get a
    zero vector rather than being dropped or crashing the encoder. Only the
    non-empty narratives are sent to the model, batched together for speed.
    """
    from sentence_transformers import SentenceTransformer

    texts = [r.get("complaint_what_happened") for r in records]
    narrative_indices = [i for i, t in enumerate(texts) if t]
    narrative_texts = [texts[i] for i in narrative_indices]

    embeddings = np.zeros((len(records), dimension), dtype=np.float32)

    if narrative_texts:
        logger.info(
            "Encoding %d/%d narratives with %s (dimension=%d); "
            "%d records with no narrative get a zero vector.",
            len(narrative_texts), len(records), model_name, dimension,
            len(records) - len(narrative_texts),
        )
        model = SentenceTransformer(model_name)
        encoded = model.encode(narrative_texts, show_progress_bar=True, convert_to_numpy=True)
        embeddings[narrative_indices] = encoded

    return embeddings


def build_text_embeddings(
    records: list[dict[str, Any]], config: dict[str, Any]
) -> pd.DataFrame:
    """Build the text-embedding matrix, indexed by complaint_id."""
    embedding_cfg = config["embedding"]
    model_name = embedding_cfg["model_name"]
    dimension = embedding_cfg["dimension"]

    embeddings = generate_embeddings(records, model_name, dimension)

    complaint_ids = [r["complaint_id"] for r in records]
    emb_df = pd.DataFrame(embeddings, columns=[f"emb_{i}" for i in range(dimension)])
    emb_df.insert(0, "complaint_id", complaint_ids)
    return emb_df


def save_embeddings(embeddings_df: pd.DataFrame, output_path: Path) -> None:
    """Write the embedding matrix to `output_path` as CSV and log its shape."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    embeddings_df.to_csv(output_path, index=False)
    n_dims = embeddings_df.shape[1] - 1  # exclude complaint_id
    logger.info(
        "Saved text embedding matrix to %s (%d rows, %d dimensions).",
        output_path, embeddings_df.shape[0], n_dims,
    )


def run_embeddings(config_path: Path | None = None) -> Path:
    """Orchestrate Section 4: load labeled data, embed narratives, save, return path."""
    config = load_config(config_path) if config_path is not None else load_config()

    labeled_path = Path(config["data"]["processed_dir"]) / config["data"]["labeled_filename"]
    records = load_labeled_data(labeled_path)

    embeddings_df = build_text_embeddings(records, config)

    output_path = (
        Path(config["data"]["processed_dir"]) / config["data"]["text_embeddings_filename"]
    )
    save_embeddings(embeddings_df, output_path)
    return output_path


def main() -> None:
    """CLI entry point: run Section 4, print a clear message and exit(1) on
    EmbeddingsError instead of a raw traceback.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        output_path = run_embeddings()
    except EmbeddingsError as e:
        print(f"Embedding build failed: {e}", file=sys.stderr)
        sys.exit(1)
    else:
        print(f"Embedding build complete: text embeddings written to {output_path}")


if __name__ == "__main__":
    main()
