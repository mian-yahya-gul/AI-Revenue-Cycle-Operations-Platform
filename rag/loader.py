"""Loads payer policy documents from disk into LangChain Document objects."""

from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document

from config.settings import PAYER_POLICY_DIR
from utils.logger import get_logger

logger = get_logger(__name__)


def load_policy_documents(policy_dir: Path = PAYER_POLICY_DIR) -> list[Document]:
    """Load every .txt policy document in the payer policy directory.

    Each file is expected to be named ``<payer>_<topic>.txt`` (e.g.
    ``aetna_prior_authorization.txt``) so the payer can be inferred for
    metadata filtering during retrieval.
    """
    documents: list[Document] = []
    if not policy_dir.exists():
        logger.warning("Policy directory does not exist: %s", policy_dir)
        return documents

    for file_path in sorted(policy_dir.glob("*.txt")):
        text = file_path.read_text(encoding="utf-8")
        payer = file_path.stem.split("_")[0].replace("-", " ").title()
        documents.append(
            Document(
                page_content=text,
                metadata={
                    "source": file_path.name,
                    "payer": payer,
                    "path": str(file_path),
                },
            )
        )
    logger.info("Loaded %d payer policy documents from %s", len(documents), policy_dir)
    return documents
