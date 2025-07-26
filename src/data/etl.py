import json
import logging
from typing import Optional
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_goodreads_data(
    file_path: str, sample_size: Optional[int] = None
) -> pd.DataFrame:
    """Load Goodreads data from JSON or JSONL format."""
    data = []

    def opener(f):
        return open(f, "r", encoding="utf-8")

    with opener(file_path) as f:
        for i, line in enumerate(f):
            if sample_size and i >= sample_size:
                break
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    df = pd.DataFrame(data)
    logger.info(f"Loaded {len(df)} records from {file_path}")
    return df


def filter_children_books(df: pd.DataFrame) -> pd.DataFrame:
    """Filter dataset for children's books."""
    if len(df) == 0:
        return df

    children_keywords = [
        "children",
        "kids",
        "picture-book",
        "middle-grade",
        "early-readers",
        "juvenile",
        "young-adult",
    ]

    # Check in popular_shelves or genres
    if "popular_shelves" in df.columns:
        mask = (
            df["popular_shelves"]
            .astype(str)
            .str.lower()
            .str.contains("|".join(children_keywords), na=False)
        )
    elif "genres" in df.columns:
        mask = (
            df["genres"]
            .astype(str)
            .str.lower()
            .str.contains("|".join(children_keywords), na=False)
        )
    else:
        # Fallback to title/description
        mask = df["title"].astype(str).str.lower().str.contains(
            "|".join(children_keywords), na=False
        ) | df["description"].astype(str).str.lower().str.contains(
            "|".join(children_keywords), na=False
        )

    filtered_df = df[mask].copy()
    logger.info(f"Filtered to {len(filtered_df)} children's books")
    return filtered_df


def preprocess_features(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess features for modeling."""
    # Handle empty dataframe
    if len(df) == 0:
        return df

    df = df.copy()

    # Ensure required columns exist with default values
    required_columns = {
        "average_rating": None,
        "num_pages": None,
        "ratings_count": 0,
        "text_reviews_count": 0,
        "title": "",
        "description": "",
        "publication_date": None,
        "publish_date": None,
    }

    for col, default_val in required_columns.items():
        if col not in df.columns:
            df[col] = default_val

    # Handle missing values for numeric columns
    df["average_rating"] = pd.to_numeric(df["average_rating"], errors="coerce")
    df["num_pages"] = pd.to_numeric(df["num_pages"], errors="coerce")
    df["ratings_count"] = pd.to_numeric(df["ratings_count"], errors="coerce")
    df["text_reviews_count"] = pd.to_numeric(df["text_reviews_count"], errors="coerce")

    # Fill missing values
    df["num_pages"] = df["num_pages"].fillna(
        df["num_pages"].median() if not df["num_pages"].isna().all() else 100
    )
    df["ratings_count"] = df["ratings_count"].fillna(0)
    df["text_reviews_count"] = df["text_reviews_count"].fillna(0)

    # Extract publication year
    if "publication_date" in df.columns and df["publication_date"].notna().any():
        df["publication_year"] = pd.to_datetime(
            df["publication_date"], errors="coerce"
        ).dt.year
    elif "publish_date" in df.columns and df["publish_date"].notna().any():
        df["publication_year"] = pd.to_datetime(
            df["publish_date"], errors="coerce"
        ).dt.year
    else:
        df["publication_year"] = 2020  # Default value

    df["publication_year"] = df["publication_year"].fillna(2020)

    # Text length features
    df["title_length"] = df["title"].astype(str).str.len()
    df["description_length"] = df["description"].astype(str).str.len()

    # Rating density (avoid division by zero)
    current_year = df["publication_year"].max() if len(df) > 0 else 2024
    years_since_pub = current_year - df["publication_year"] + 1
    years_since_pub = years_since_pub.clip(lower=1)  # Ensure at least 1 year
    df["rating_density"] = df["ratings_count"] / years_since_pub

    return df
