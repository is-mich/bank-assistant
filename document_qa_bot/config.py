"""
Configuration for Document Q&A Bot
===================================
"""

from pathlib import Path
from dataclasses import dataclass
from functools import lru_cache


@dataclass
class Config:
    """Application configuration"""

    # Experiment
    experiment_name: str = "default"
    collection_name: str = "documents"

    # Paths
    base_path: Path = Path(__file__).parent
    chroma_db_path: Path = None
    sample_docs_path: Path = None
    logs_path: Path = None  # interaction logs of the user app; contains user input, never commit

    # Chunking
    chunking_strategy: str = "recursive"  # "recursive" (fixed-size) or "structure" (chapters/clauses)
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # Retrieval
    top_k: int = 8
    min_relevance_score: float = 0.15

    # Query expansion (Standard mode): extra sub-queries, merged and capped
    query_expansion: bool = True
    num_subqueries: int = 4
    max_context_chunks: int = 12

    # Embedding
    embedding_model: str = "text-embedding-3-small"

    # Generation
    llm_model: str = "gpt-4o-mini"
    max_tokens: int = 1500
    temperature: float = 0.3

    # UI only: pause per streamed token so the answer appears gradually
    stream_delay_seconds: float = 0.03
    # User app: requests the Gradio queue processes at the same time
    ui_concurrency: int = 4

    # Evaluation: judge model and token prices in USD per 1M tokens (verify against the provider's price list)
    judge_model: str = "gpt-4o-mini"
    price_per_1m_input_tokens: float = 0.15
    price_per_1m_output_tokens: float = 0.60

    def __post_init__(self):
        if self.chroma_db_path is None:
            self.chroma_db_path = self.base_path / "chroma_db" / self.experiment_name
        if self.sample_docs_path is None:
            self.sample_docs_path = self.base_path / "sample_docs"
        if self.logs_path is None:
            self.logs_path = self.base_path.parent / "logs"

        # Create directories if they don't exist
        self.chroma_db_path.mkdir(parents=True, exist_ok=True)
        self.sample_docs_path.mkdir(exist_ok=True)

    def use_experiment(self, name: str) -> None:
        """Switch to another experiment; each one gets its own Chroma directory."""
        self.experiment_name = name
        self.chroma_db_path = self.base_path / "chroma_db" / name
        self.chroma_db_path.mkdir(parents=True, exist_ok=True)


@lru_cache()
def get_config() -> Config:
    """Get cached configuration instance"""
    return Config()
