import pandas as pd


class LLMFeatureEnhancer:
    """Use LLM-like analysis for feature enhancement."""

    def __init__(self):
        self.quality_keywords = self._load_quality_keywords()

    def _load_quality_keywords(self):
        """Load quality indicators for children's books."""
        return {
            "high_quality": [
                "award",
                "winner",
                "caldecott",
                "newbery",
                "bestseller",
                "classic",
                "beloved",
                "acclaimed",
                "timeless",
                "masterpiece",
                "wonderful",
                "delightful",
                "charming",
                "magical",
                "captivating",
                "engaging",
            ],
            "educational_value": [
                "learn",
                "teach",
                "educational",
                "curriculum",
                "lesson",
                "skill",
                "alphabet",
                "numbers",
                "counting",
                "reading",
                "vocabulary",
                "science",
                "history",
                "geography",
                "moral",
                "values",
            ],
            "entertainment_value": [
                "fun",
                "funny",
                "hilarious",
                "adventure",
                "exciting",
                "thrilling",
                "mystery",
                "surprise",
                "humor",
                "laugh",
                "giggle",
                "silly",
                "playful",
                "imaginative",
                "creative",
            ],
            "age_appropriateness": [
                "age appropriate",
                "suitable for",
                "perfect for",
                "ideal for",
                "recommended for",
                "designed for",
                "tailored to",
            ],
            "negative_indicators": [
                "boring",
                "dull",
                "confusing",
                "difficult",
                "inappropriate",
                "scary",
                "violent",
                "complex",
                "advanced",
                "challenging",
            ],
        }

    def analyze_book_quality(self, df):
        """Analyze book quality using keyword analysis."""
        df = df.copy()

        # Combine title and description
        df["full_text"] = (
            df["title"].astype(str) + " " + df["description"].astype(str)
        ).str.lower()

        # Quality score based on keywords
        for category, keywords in self.quality_keywords.items():
            df[f"{category}_score"] = (
                df["full_text"]
                .apply(lambda x: sum(keyword in str(x) for keyword in keywords))
                .astype(float)
            )  # Ensure numeric

        # Overall quality score
        df["llm_quality_score"] = (
            df["high_quality_score"] * 3
            + df["educational_value_score"] * 2
            + df["entertainment_value_score"] * 2
            + df["age_appropriateness_score"] * 1
            - df["negative_indicators_score"] * 2
        ).astype(float)

        # Normalize by text length
        text_length = df["full_text"].str.len().fillna(1)
        df["normalized_quality_score"] = (
            df["llm_quality_score"] / (text_length + 1) * 1000
        ).astype(float)

        return df

    def extract_narrative_features(self, df):
        """Extract narrative structure features."""
        df = df.copy()

        # Story structure indicators
        story_elements = {
            "character_focus": ["character", "protagonist", "hero", "friend", "family"],
            "plot_elements": ["story", "adventure", "journey", "quest", "mystery"],
            "setting_richness": [
                "world",
                "place",
                "land",
                "kingdom",
                "forest",
                "ocean",
            ],
            "emotional_content": [
                "feel",
                "emotion",
                "happy",
                "sad",
                "excited",
                "worried",
            ],
            "interactive_elements": ["ask", "question", "think", "imagine", "discover"],
        }

        full_text = (
            df["title"].astype(str) + " " + df["description"].astype(str)
        ).str.lower()

        for element, keywords in story_elements.items():
            df[f"{element}_score"] = full_text.apply(
                lambda x: sum(keyword in str(x) for keyword in keywords)
            ).astype(float)  # Ensure numeric

        # Narrative complexity score
        df["narrative_complexity"] = (
            df["character_focus_score"]
            + df["plot_elements_score"]
            + df["setting_richness_score"]
            + df["emotional_content_score"]
            + df["interactive_elements_score"]
        ).astype(float)

        return df

    def extract_market_features(self, df):
        """Extract market positioning features."""
        df = df.copy()

        # Series and franchise indicators
        series_patterns = [
            r"book \d+",
            r"#\d+",
            r"volume \d+",
            r"part \d+",
            "series",
            "collection",
            "set",
        ]

        df["is_part_of_series"] = 0
        title_lower = df["title"].astype(str).str.lower()

        for pattern in series_patterns:
            df["is_part_of_series"] += title_lower.str.contains(
                pattern, regex=True, na=False
            ).astype(int)

        df["is_part_of_series"] = (df["is_part_of_series"] > 0).astype(int)

        # Format indicators
        format_keywords = {
            "picture_book": ["picture book", "illustrated", "board book"],
            "chapter_book": ["chapter book", "early reader", "beginning reader"],
            "graphic_novel": ["graphic", "comic", "manga"],
            "activity_book": ["activity", "coloring", "puzzle", "workbook"],
        }

        full_text = (
            df["title"].astype(str) + " " + df["description"].astype(str)
        ).str.lower()

        for format_type, keywords in format_keywords.items():
            df[f"is_{format_type}"] = full_text.apply(
                lambda x: int(any(keyword in str(x) for keyword in keywords))
            ).astype(int)  # Ensure numeric

        return df

    def build_llm_features(self, df):
        """Build all LLM-inspired features."""
        df = df.copy()

        # Apply all analysis methods
        df = self.analyze_book_quality(df)
        df = self.extract_narrative_features(df)
        df = self.extract_market_features(df)

        # Select only the new features (exclude full_text and other non-numeric columns)
        llm_features = [
            col
            for col in df.columns
            if col.endswith("_score")
            or col.startswith("is_")
            or "quality" in col
            or "complexity" in col
        ]

        feature_df = df[llm_features].copy()

        # Ensure all features are numeric
        for col in feature_df.columns:
            feature_df[col] = pd.to_numeric(feature_df[col], errors="coerce").fillna(0)

        return feature_df
