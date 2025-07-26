import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.feature_selection import SelectKBest, f_regression
import re


class AdvancedFeatureBuilder:
    """Advanced feature engineering specifically for book ratings."""

    def __init__(self):
        self.scaler = StandardScaler()
        self.quantile_transformer = QuantileTransformer(output_distribution="normal")
        self.feature_selector = SelectKBest(f_regression, k=50)
        self.is_fitted = False

    def create_rating_features(self, df):
        """Create features that correlate with book quality."""
        df = df.copy()

        # Rating engagement features
        df["rating_engagement"] = df["ratings_count"] / (df["text_reviews_count"] + 1)
        df["review_engagement"] = df["text_reviews_count"] / (df["ratings_count"] + 1)

        # Popularity buckets
        df["popularity_percentile"] = pd.qcut(
            df["ratings_count"], q=10, labels=False, duplicates="drop"
        )
        df["is_highly_rated_count"] = (
            df["ratings_count"] > df["ratings_count"].quantile(0.8)
        ).astype(int)
        df["is_niche_book"] = (
            df["ratings_count"] < df["ratings_count"].quantile(0.2)
        ).astype(int)

        # Age and recency effects
        current_year = 2024
        df["years_since_pub"] = current_year - df["publication_year"]
        df["is_golden_age"] = (
            (df["publication_year"] >= 1950) & (df["publication_year"] <= 1990)
        ).astype(int)
        df["is_digital_age"] = (df["publication_year"] >= 2000).astype(int)

        # Page count quality indicators
        df["pages_per_rating"] = df["num_pages"] / (df["ratings_count"] + 1)
        df["optimal_length"] = (
            (df["num_pages"] >= 24) & (df["num_pages"] <= 48)
        ).astype(int)  # Sweet spot for picture books

        return df

    def create_text_quality_features(self, df):
        """Extract text quality indicators."""
        df = df.copy()

        # Title quality features
        df["title_readability"] = (
            df["title"].astype(str).apply(self._calculate_readability)
        )
        df["title_sentiment"] = df["title"].astype(str).apply(self._calculate_sentiment)
        df["title_complexity"] = (
            df["title"].astype(str).apply(lambda x: len(set(x.lower().split())))
        )

        # Description quality features
        df["desc_readability"] = (
            df["description"].astype(str).apply(self._calculate_readability)
        )
        df["desc_sentiment"] = (
            df["description"].astype(str).apply(self._calculate_sentiment)
        )
        df["desc_complexity"] = (
            df["description"]
            .astype(str)
            .apply(lambda x: len(set(x.lower().split())) / (len(x.split()) + 1))
        )

        # Quality keywords
        quality_words = [
            "award",
            "bestselling",
            "acclaimed",
            "beloved",
            "classic",
            "timeless",
            "wonderful",
        ]
        df["quality_word_count"] = (
            df["description"]
            .astype(str)
            .str.lower()
            .apply(lambda x: sum(word in x for word in quality_words))
        )

        # Age appropriateness
        age_indicators = {
            "baby": 0.5,
            "toddler": 1.5,
            "preschool": 3,
            "kindergarten": 5,
            "first grade": 6,
            "second grade": 7,
            "chapter book": 8,
            "middle grade": 10,
            "young adult": 14,
        }

        df["estimated_age"] = df["title"].astype(str).str.lower()
        for indicator, age in age_indicators.items():
            df.loc[
                df["estimated_age"].str.contains(indicator, na=False), "estimated_age"
            ] = age

        df["estimated_age"] = pd.to_numeric(
            df["estimated_age"], errors="coerce"
        ).fillna(6)  # Default to 6 years old

        return df

    def _calculate_readability(self, text):
        """Simple readability score based on word and sentence length."""
        if not text or pd.isna(text):
            return 0

        words = text.split()
        sentences = re.split(r"[.!?]+", text)

        if len(sentences) == 0 or len(words) == 0:
            return 0

        avg_word_length = np.mean([len(word) for word in words])
        avg_sentence_length = len(words) / len(sentences)

        # Simple readability formula (lower is easier)
        readability = avg_word_length + (avg_sentence_length / 10)
        return readability

    def _calculate_sentiment(self, text):
        """Simple sentiment analysis based on positive/negative words."""
        if not text or pd.isna(text):
            return 0

        positive_words = [
            "good",
            "great",
            "excellent",
            "wonderful",
            "amazing",
            "fantastic",
            "love",
            "beautiful",
            "fun",
            "exciting",
        ]
        negative_words = [
            "bad",
            "terrible",
            "awful",
            "boring",
            "difficult",
            "confusing",
            "sad",
            "scary",
            "inappropriate",
        ]

        text_lower = text.lower()
        positive_count = sum(word in text_lower for word in positive_words)
        negative_count = sum(word in text_lower for word in negative_words)

        total_words = len(text.split())
        if total_words == 0:
            return 0

        return (positive_count - negative_count) / total_words

    def create_interaction_features(self, df):
        """Create interaction features between important variables."""
        df = df.copy()

        # Key interactions
        df["pages_age_interaction"] = df["num_pages"] * df["years_since_pub"]
        df["popularity_age_interaction"] = df["ratings_count"] * df["years_since_pub"]
        df["quality_popularity"] = df["quality_word_count"] * np.log1p(
            df["ratings_count"]
        )
        df["sentiment_popularity"] = df.get("desc_sentiment", 0) * np.log1p(
            df["ratings_count"]
        )

        # Ratio features
        df["title_desc_length_ratio"] = df["title_length"] / (
            df["description_length"] + 1
        )
        df["pages_to_words_ratio"] = df["num_pages"] / (
            df.get("desc_word_count", 100) + 1
        )

        return df

    def build_features(self, df, is_training=True):
        """Build all advanced features."""
        df = df.copy()

        # Apply all feature engineering steps
        df = self.create_rating_features(df)
        df = self.create_text_quality_features(df)
        df = self.create_interaction_features(df)

        # Select numeric features
        numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
        if "average_rating" in numeric_features:
            numeric_features.remove("average_rating")

        feature_df = df[numeric_features].fillna(0)

        # Transform features
        if is_training:
            # Use quantile transformer to make features more normal
            feature_df_transformed = pd.DataFrame(
                self.quantile_transformer.fit_transform(feature_df),
                columns=feature_df.columns,
                index=feature_df.index,
            )

            # Feature selection
            if "average_rating" in df.columns:
                self.feature_selector.fit(feature_df_transformed, df["average_rating"])

            self.is_fitted = True
        else:
            feature_df_transformed = pd.DataFrame(
                self.quantile_transformer.transform(feature_df),
                columns=feature_df.columns,
                index=feature_df.index,
            )

        # Apply feature selection
        if hasattr(self.feature_selector, "transform"):
            selected_features = self.feature_selector.transform(feature_df_transformed)
            selected_feature_names = feature_df.columns[
                self.feature_selector.get_support()
            ]

            feature_df_final = pd.DataFrame(
                selected_features,
                columns=selected_feature_names,
                index=feature_df.index,
            )
        else:
            feature_df_final = feature_df_transformed

        print(
            f"Built {feature_df_final.shape[1]} features for {len(feature_df_final)} samples"
        )
        return feature_df_final
