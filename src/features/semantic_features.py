import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle
from pathlib import Path
import logging

# For embeddings - we'll use sentence-transformers
try:
    from sentence_transformers import SentenceTransformer

    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    print("Install sentence-transformers: pip install sentence-transformers")

logger = logging.getLogger(__name__)


class SemanticFeatureBuilder:
    """Build semantic features using embeddings and similarity search."""

    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.embedding_model = None
        self.scaler = StandardScaler()
        self.book_embeddings = None
        self.rating_database = None
        self.is_fitted = False

        if EMBEDDINGS_AVAILABLE:
            self.embedding_model = SentenceTransformer(model_name)

    def create_book_text(self, df):
        """Combine title and description for embedding."""
        df = df.copy()

        # Create comprehensive text representation
        df["book_text"] = (
            df["title"].astype(str) + ". " + df["description"].astype(str)
        ).str.strip()

        # Clean the text
        df["book_text"] = df["book_text"].str.replace(r"[^\w\s.]", " ", regex=True)
        df["book_text"] = df["book_text"].str.replace(r"\s+", " ", regex=True)

        return df

    def extract_semantic_features(self, df, is_training=True):
        """Extract semantic features using embeddings."""
        if not EMBEDDINGS_AVAILABLE:
            print("Sentence transformers not available, skipping semantic features")
            return pd.DataFrame(index=df.index)

        df = self.create_book_text(df)

        # Generate embeddings
        print("Generating embeddings...")
        embeddings = self.embedding_model.encode(
            df["book_text"].tolist(), show_progress_bar=True, batch_size=32
        )

        # Create embedding features
        embedding_df = pd.DataFrame(
            embeddings,
            columns=[f"embedding_{i}" for i in range(embeddings.shape[1])],
            index=df.index,
        )

        if is_training:
            # Store embeddings and ratings for similarity search
            self.book_embeddings = embeddings
            if "average_rating" in df.columns:
                self.rating_database = df["average_rating"].values

            # Fit scaler on embeddings
            self.scaler.fit(embeddings)
            self.is_fitted = True

        # Scale embeddings
        scaled_embeddings = self.scaler.transform(embeddings)
        embedding_df = pd.DataFrame(
            scaled_embeddings, columns=embedding_df.columns, index=df.index
        )

        return embedding_df

    def find_similar_books(self, query_embeddings, top_k=10):
        """Find similar books using cosine similarity."""
        if self.book_embeddings is None:
            return np.array([])

        from sklearn.metrics.pairwise import cosine_similarity

        similarities = cosine_similarity(query_embeddings, self.book_embeddings)

        # Get top-k similar books for each query
        similar_ratings = []
        for sim_row in similarities:
            top_indices = np.argsort(sim_row)[-top_k:]
            if self.rating_database is not None:
                similar_book_ratings = self.rating_database[top_indices]
                similar_ratings.append(
                    {
                        "mean_similar_rating": np.mean(similar_book_ratings),
                        "std_similar_rating": np.std(similar_book_ratings),
                        "median_similar_rating": np.median(similar_book_ratings),
                        "max_similarity": np.max(sim_row),
                        "mean_similarity": np.mean(sim_row[top_indices]),
                    }
                )
            else:
                similar_ratings.append(
                    {
                        "mean_similar_rating": 0,
                        "std_similar_rating": 0,
                        "median_similar_rating": 0,
                        "max_similarity": np.max(sim_row),
                        "mean_similarity": 0,
                    }
                )

        return pd.DataFrame(similar_ratings)

    def build_semantic_features(self, df, is_training=True):
        """Build all semantic features."""
        features = []

        # 1. Embeddings
        embedding_features = self.extract_semantic_features(df, is_training)
        if not embedding_features.empty:
            features.append(embedding_features)

        # 2. Similarity-based features
        if not embedding_features.empty and self.is_fitted:
            similarity_features = self.find_similar_books(
                embedding_features.values, top_k=20
            )
            similarity_features.index = df.index
            features.append(similarity_features)

        # 3. Text quality features using embeddings
        if not embedding_features.empty:
            text_quality = self.extract_text_quality_features(df)
            features.append(text_quality)

        # Combine all features
        if features:
            result = pd.concat(features, axis=1)
        else:
            result = pd.DataFrame(index=df.index)

        logger.info(
            f"Built {result.shape[1]} semantic features for {len(result)} samples"
        )
        return result

    def extract_text_quality_features(self, df):
        """Extract text quality features using semantic analysis."""
        df = self.create_book_text(df)

        # Quality indicators based on text analysis
        quality_features = pd.DataFrame(index=df.index)

        # Text complexity (vocabulary diversity)
        quality_features["vocab_diversity"] = df["book_text"].apply(
            lambda x: len(set(x.lower().split())) / (len(x.split()) + 1)
        )

        # Emotional words detection
        positive_words = [
            "amazing",
            "wonderful",
            "fantastic",
            "delightful",
            "charming",
            "magical",
            "adventure",
            "fun",
        ]
        negative_words = [
            "boring",
            "confusing",
            "difficult",
            "scary",
            "inappropriate",
            "dull",
        ]

        quality_features["positive_word_density"] = (
            df["book_text"]
            .str.lower()
            .apply(
                lambda x: sum(word in x for word in positive_words)
                / (len(x.split()) + 1)
            )
        )

        quality_features["negative_word_density"] = (
            df["book_text"]
            .str.lower()
            .apply(
                lambda x: sum(word in x for word in negative_words)
                / (len(x.split()) + 1)
            )
        )

        # Age appropriateness indicators
        age_words = {
            "baby": 1,
            "toddler": 2,
            "preschool": 4,
            "kindergarten": 5,
            "first grade": 6,
            "second grade": 7,
            "elementary": 8,
            "middle grade": 10,
            "young adult": 14,
        }

        quality_features["estimated_age_level"] = 6  # Default
        for age_term, age_val in age_words.items():
            mask = df["book_text"].str.lower().str.contains(age_term, na=False)
            quality_features.loc[mask, "estimated_age_level"] = age_val

        return quality_features

    def save_semantic_model(self, path: str):
        """Save semantic model components."""
        Path(path).mkdir(parents=True, exist_ok=True)

        # Save scaler
        with open(f"{path}/semantic_scaler.pkl", "wb") as f:
            pickle.dump(self.scaler, f)

        # Save embeddings database
        if self.book_embeddings is not None:
            np.save(f"{path}/book_embeddings.npy", self.book_embeddings)

        if self.rating_database is not None:
            np.save(f"{path}/rating_database.npy", self.rating_database)

        # Save metadata
        metadata = {"model_name": self.model_name, "is_fitted": self.is_fitted}
        with open(f"{path}/semantic_metadata.pkl", "wb") as f:
            pickle.dump(metadata, f)

        logger.info(f"Saved semantic model to {path}")

    def load_semantic_model(self, path: str):
        """Load semantic model components."""
        # Load scaler
        with open(f"{path}/semantic_scaler.pkl", "rb") as f:
            self.scaler = pickle.load(f)

        # Load embeddings database
        if Path(f"{path}/book_embeddings.npy").exists():
            self.book_embeddings = np.load(f"{path}/book_embeddings.npy")

        if Path(f"{path}/rating_database.npy").exists():
            self.rating_database = np.load(f"{path}/rating_database.npy")

        # Load metadata
        with open(f"{path}/semantic_metadata.pkl", "rb") as f:
            metadata = pickle.load(f)
            self.model_name = metadata["model_name"]
            self.is_fitted = metadata["is_fitted"]

        # Reinitialize embedding model
        if EMBEDDINGS_AVAILABLE:
            self.embedding_model = SentenceTransformer(self.model_name)

        logger.info(f"Loaded semantic model from {path}")
