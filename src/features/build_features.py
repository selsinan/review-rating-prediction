import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures
from sklearn.decomposition import TruncatedSVD
import pickle
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class FeatureBuilder:
    """Enhanced feature builder with domain-specific features for children's books."""

    def __init__(self):
        self.tfidf_title = TfidfVectorizer(
            max_features=50,
            stop_words="english",
            lowercase=True,
            ngram_range=(1, 2),
            min_df=2,
        )
        self.tfidf_desc = TfidfVectorizer(
            max_features=100,
            stop_words="english",
            lowercase=True,
            ngram_range=(1, 2),
            min_df=2,
        )
        self.svd_title = TruncatedSVD(n_components=10, random_state=42)
        self.svd_desc = TruncatedSVD(n_components=20, random_state=42)
        self.scaler = StandardScaler()
        self.poly = PolynomialFeatures(
            degree=2, interaction_only=True, include_bias=False
        )
        self.label_encoders = {}
        self.is_fitted = False

    def extract_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract advanced domain-specific features."""
        df = df.copy()

        # Title analysis
        df["title_word_count"] = df["title"].astype(str).apply(lambda x: len(x.split()))
        df["title_char_count"] = df["title"].astype(str).str.len()
        df["title_has_number"] = (
            df["title"].astype(str).str.contains(r"\d+", na=False).astype(int)
        )
        df["title_has_colon"] = (
            df["title"].astype(str).str.contains(":", na=False).astype(int)
        )
        df["title_has_series"] = (
            df["title"]
            .astype(str)
            .str.contains(r"#\d+|Book \d+|Volume \d+", na=False)
            .astype(int)
        )
        df["title_exclamation"] = df["title"].astype(str).str.count("!")
        df["title_question"] = df["title"].astype(str).str.count(r"\?")
        df["title_uppercase_ratio"] = (
            df["title"]
            .astype(str)
            .apply(lambda x: sum(1 for c in x if c.isupper()) / (len(x) + 1))
        )

        # Age-related keywords in title
        age_keywords = [
            "baby",
            "toddler",
            "preschool",
            "kindergarten",
            "grade",
            "teen",
            "young",
        ]
        df["title_has_age"] = (
            df["title"]
            .astype(str)
            .str.lower()
            .str.contains("|".join(age_keywords), na=False)
            .astype(int)
        )

        # Description analysis
        df["desc_word_count"] = (
            df["description"].astype(str).apply(lambda x: len(x.split()))
        )
        df["desc_char_count"] = df["description"].astype(str).str.len()
        df["desc_sentence_count"] = df["description"].astype(str).str.count(r"[.!?]+")
        df["desc_avg_word_length"] = (
            df["description"]
            .astype(str)
            .apply(
                lambda x: np.mean([len(word) for word in x.split()]) if x.split() else 0
            )
        )

        # Award and quality indicators
        award_keywords = [
            "award",
            "winner",
            "medal",
            "prize",
            "bestseller",
            "caldecott",
            "newbery",
        ]
        df["desc_has_award"] = (
            df["description"]
            .astype(str)
            .str.lower()
            .str.contains("|".join(award_keywords), na=False)
            .astype(int)
        )

        # Reading level indicators
        level_keywords = [
            "grade",
            "age",
            "years old",
            "kindergarten",
            "elementary",
            "reading level",
        ]
        df["desc_has_reading_level"] = (
            df["description"]
            .astype(str)
            .str.lower()
            .str.contains("|".join(level_keywords), na=False)
            .astype(int)
        )

        # Emotional/educational keywords
        positive_keywords = [
            "fun",
            "adventure",
            "magical",
            "wonderful",
            "amazing",
            "delightful",
        ]
        educational_keywords = [
            "learn",
            "teach",
            "educational",
            "lesson",
            "moral",
            "values",
        ]

        df["desc_positive_words"] = (
            df["description"]
            .astype(str)
            .str.lower()
            .str.count("|".join(positive_keywords))
        )
        df["desc_educational_words"] = (
            df["description"]
            .astype(str)
            .str.lower()
            .str.count("|".join(educational_keywords))
        )

        # Publication and popularity features
        current_year = 2024
        df["book_age"] = current_year - df["publication_year"]
        df["book_age_squared"] = df["book_age"] ** 2
        df["is_very_recent"] = (df["book_age"] <= 2).astype(int)
        df["is_recent"] = ((df["book_age"] > 2) & (df["book_age"] <= 5)).astype(int)
        df["is_modern"] = ((df["book_age"] > 5) & (df["book_age"] <= 15)).astype(int)
        df["is_classic"] = (df["book_age"] > 20).astype(int)

        # Rating and review features
        df["log_ratings_count"] = np.log1p(df["ratings_count"])
        df["log_reviews_count"] = np.log1p(df["text_reviews_count"])
        df["review_ratio"] = df["text_reviews_count"] / (df["ratings_count"] + 1)
        df["popularity_score"] = df["log_ratings_count"] * df["log_reviews_count"]

        # Page count features for children's books
        df["log_pages"] = np.log1p(df["num_pages"])
        df["is_picture_book"] = (df["num_pages"] <= 32).astype(int)
        df["is_early_reader"] = (
            (df["num_pages"] > 32) & (df["num_pages"] <= 64)
        ).astype(int)
        df["is_chapter_book"] = (
            (df["num_pages"] > 64) & (df["num_pages"] <= 200)
        ).astype(int)
        df["is_middle_grade"] = (
            (df["num_pages"] > 200) & (df["num_pages"] <= 400)
        ).astype(int)
        df["is_long_book"] = (df["num_pages"] > 400).astype(int)

        # Derived ratios and interactions
        df["pages_per_year"] = df["num_pages"] / (df["book_age"] + 1)
        df["ratings_per_year"] = df["ratings_count"] / (df["book_age"] + 1)
        df["title_desc_ratio"] = df["title_char_count"] / (df["desc_char_count"] + 1)

        return df

    def build_features(
        self, df: pd.DataFrame, is_training: bool = True
    ) -> pd.DataFrame:
        """Build comprehensive features for modeling."""
        df = df.copy()

        # Ensure required columns exist
        required_cols = ["title", "description"]
        for col in required_cols:
            if col not in df.columns:
                df[col] = ""

        # Extract advanced features
        df = self.extract_advanced_features(df)

        # Base numerical features
        base_numerical = [
            "num_pages",
            "ratings_count",
            "text_reviews_count",
            "publication_year",
            "title_length",
            "description_length",
            "rating_density",
        ]

        # Advanced numerical features
        advanced_numerical = [
            "title_word_count",
            "title_char_count",
            "title_has_number",
            "title_has_colon",
            "title_has_series",
            "title_exclamation",
            "title_question",
            "title_uppercase_ratio",
            "title_has_age",
            "desc_word_count",
            "desc_char_count",
            "desc_sentence_count",
            "desc_avg_word_length",
            "desc_has_award",
            "desc_has_reading_level",
            "desc_positive_words",
            "desc_educational_words",
            "book_age",
            "book_age_squared",
            "is_very_recent",
            "is_recent",
            "is_modern",
            "is_classic",
            "log_ratings_count",
            "log_reviews_count",
            "review_ratio",
            "popularity_score",
            "log_pages",
            "is_picture_book",
            "is_early_reader",
            "is_chapter_book",
            "is_middle_grade",
            "is_long_book",
            "pages_per_year",
            "ratings_per_year",
            "title_desc_ratio",
        ]

        numerical_features = base_numerical + advanced_numerical

        # Ensure all features exist
        for feature in numerical_features:
            if feature not in df.columns:
                df[feature] = 0

        # Handle categorical features
        categorical_features = []
        if "language_code" in df.columns:
            categorical_features.append("language_code")
            df["language_code"] = df["language_code"].fillna("en")

        # Text features with SVD
        title_text = df["title"].fillna("").astype(str)
        desc_text = df["description"].fillna("").astype(str)

        if is_training:
            # TF-IDF
            title_tfidf = self.tfidf_title.fit_transform(title_text)
            desc_tfidf = self.tfidf_desc.fit_transform(desc_text)

            # SVD for dimensionality reduction
            title_svd = self.svd_title.fit_transform(title_tfidf)
            desc_svd = self.svd_desc.fit_transform(desc_tfidf)

            self.is_fitted = True
        else:
            if not self.is_fitted:
                raise ValueError(
                    "FeatureBuilder must be fitted before transforming test data"
                )
            title_tfidf = self.tfidf_title.transform(title_text)
            desc_tfidf = self.tfidf_desc.transform(desc_text)
            title_svd = self.svd_title.transform(title_tfidf)
            desc_svd = self.svd_desc.transform(desc_tfidf)

        # Create feature dataframe
        feature_df = df[numerical_features].copy()

        # Add SVD features (more compact than full TF-IDF)
        title_svd_features = pd.DataFrame(
            title_svd,
            columns=[f"title_svd_{i}" for i in range(title_svd.shape[1])],
            index=df.index,
        )
        desc_svd_features = pd.DataFrame(
            desc_svd,
            columns=[f"desc_svd_{i}" for i in range(desc_svd.shape[1])],
            index=df.index,
        )

        feature_df = pd.concat(
            [feature_df, title_svd_features, desc_svd_features], axis=1
        )

        # Handle categorical features
        for cat_feature in categorical_features:
            if is_training:
                self.label_encoders[cat_feature] = LabelEncoder()
                feature_df[f"{cat_feature}_encoded"] = self.label_encoders[
                    cat_feature
                ].fit_transform(df[cat_feature])
            else:
                le = self.label_encoders[cat_feature]
                transformed = []
                for val in df[cat_feature]:
                    if val in le.classes_:
                        transformed.append(le.transform([val])[0])
                    else:
                        transformed.append(-1)
                feature_df[f"{cat_feature}_encoded"] = transformed

        # Scale numerical features
        if is_training:
            feature_df[numerical_features] = self.scaler.fit_transform(
                feature_df[numerical_features]
            )
        else:
            feature_df[numerical_features] = self.scaler.transform(
                feature_df[numerical_features]
            )

        # Add polynomial features for key interactions (selective)
        key_features = [
            "log_ratings_count",
            "log_reviews_count",
            "log_pages",
            "book_age",
            "popularity_score",
        ]

        if is_training:
            poly_features = self.poly.fit_transform(feature_df[key_features])
            poly_feature_names = self.poly.get_feature_names_out(key_features)
        else:
            poly_features = self.poly.transform(feature_df[key_features])
            poly_feature_names = self.poly.get_feature_names_out(key_features)

        poly_df = pd.DataFrame(
            poly_features, columns=poly_feature_names, index=df.index
        )

        # Remove original features from poly_df to avoid duplication
        poly_df = poly_df.drop(columns=key_features, errors="ignore")

        feature_df = pd.concat([feature_df, poly_df], axis=1)

        logger.info(
            f"Built {feature_df.shape[1]} features for {len(feature_df)} samples"
        )
        return feature_df

    def save_preprocessors(self, path: str):
        """Save fitted preprocessors."""
        Path(path).mkdir(parents=True, exist_ok=True)

        with open(f"{path}/tfidf_title.pkl", "wb") as f:
            pickle.dump(self.tfidf_title, f)
        with open(f"{path}/tfidf_desc.pkl", "wb") as f:
            pickle.dump(self.tfidf_desc, f)
        with open(f"{path}/svd_title.pkl", "wb") as f:
            pickle.dump(self.svd_title, f)
        with open(f"{path}/svd_desc.pkl", "wb") as f:
            pickle.dump(self.svd_desc, f)
        with open(f"{path}/scaler.pkl", "wb") as f:
            pickle.dump(self.scaler, f)
        with open(f"{path}/poly.pkl", "wb") as f:
            pickle.dump(self.poly, f)
        with open(f"{path}/label_encoders.pkl", "wb") as f:
            pickle.dump(self.label_encoders, f)

        logger.info(f"Saved preprocessors to {path}")

    def load_preprocessors(self, path: str):
        """Load fitted preprocessors."""
        with open(f"{path}/tfidf_title.pkl", "rb") as f:
            self.tfidf_title = pickle.load(f)
        with open(f"{path}/tfidf_desc.pkl", "rb") as f:
            self.tfidf_desc = pickle.load(f)
        with open(f"{path}/svd_title.pkl", "rb") as f:
            self.svd_title = pickle.load(f)
        with open(f"{path}/svd_desc.pkl", "rb") as f:
            self.svd_desc = pickle.load(f)
        with open(f"{path}/scaler.pkl", "rb") as f:
            self.scaler = pickle.load(f)
        with open(f"{path}/poly.pkl", "rb") as f:
            self.poly = pickle.load(f)
        with open(f"{path}/label_encoders.pkl", "rb") as f:
            self.label_encoders = pickle.load(f)

        self.is_fitted = True
        logger.info(f"Loaded preprocessors from {path}")
