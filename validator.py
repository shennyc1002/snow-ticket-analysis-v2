"""Validation layer for ensuring data integrity in production."""

import pandas as pd
from typing import Optional
import config


class ValidationError(Exception):
    """Raised when validation fails."""
    pass


class DataValidator:
    """Validates data integrity at every stage of processing."""

    def __init__(self, original_df: pd.DataFrame):
        self.original_df = original_df.copy()
        self.original_row_count = len(original_df)
        self.original_columns = set(original_df.columns)
        self.validation_log: list[str] = []

    def log(self, message: str) -> None:
        """Add a message to the validation log."""
        self.validation_log.append(message)
        print(f"  [Validator] {message}")

    def validate_row_count(self, df: pd.DataFrame, stage: str) -> None:
        """Ensure no rows were lost during processing."""
        current_count = len(df)
        if current_count != self.original_row_count:
            raise ValidationError(
                f"Row count mismatch at stage '{stage}': "
                f"Expected {self.original_row_count}, got {current_count}"
            )
        self.log(f"[OK] Row count validated at '{stage}': {current_count} rows")

    def validate_original_data_unchanged(self, df: pd.DataFrame, stage: str) -> None:
        """Ensure original columns weren't modified."""
        for col in self.original_columns:
            if col not in df.columns:
                raise ValidationError(f"Original column '{col}' missing at stage '{stage}'")

            # Compare values (handling NaN)
            original_values = self.original_df[col].fillna("__NA__").tolist()
            current_values = df[col].fillna("__NA__").tolist()

            if original_values != current_values:
                raise ValidationError(
                    f"Original column '{col}' was modified at stage '{stage}'"
                )

        self.log(f"[OK] Original data integrity validated at '{stage}'")

    def validate_categories(self, categories: dict[int, str]) -> None:
        """Validate the category assignments."""
        # Check all indices are covered
        expected_indices = set(range(self.original_row_count))
        actual_indices = set(categories.keys())

        missing = expected_indices - actual_indices
        if missing:
            raise ValidationError(f"Missing category assignments for indices: {sorted(missing)[:10]}...")

        extra = actual_indices - expected_indices
        if extra:
            raise ValidationError(f"Unexpected category indices: {sorted(extra)[:10]}...")

        # Check no empty categories
        empty_cats = [idx for idx, cat in categories.items() if not cat or not cat.strip()]
        if empty_cats:
            raise ValidationError(f"Empty categories found for indices: {empty_cats[:10]}...")

        # Check category names are reasonable
        for idx, cat in categories.items():
            if len(cat) > 100:
                raise ValidationError(f"Category too long for index {idx}: {cat[:50]}...")
            if len(cat) < 2:
                raise ValidationError(f"Category too short for index {idx}: '{cat}'")

        self.log(f"[OK] Categories validated: {len(set(categories.values()))} unique categories for {len(categories)} tickets")

    def validate_final_output(self, df: pd.DataFrame) -> None:
        """Final validation before saving."""
        self.validate_row_count(df, "final_output")
        self.validate_original_data_unchanged(df, "final_output")

        # Check category column exists
        cat_col = config.COLUMN_MAPPING["category"]
        if cat_col not in df.columns:
            raise ValidationError(f"Category column '{cat_col}' not found in output")

        # Check no null categories
        null_count = df[cat_col].isna().sum()
        if null_count > 0:
            raise ValidationError(f"Found {null_count} null values in category column")

        # Check category distribution
        category_counts = df[cat_col].value_counts()
        total_categories = len(category_counts)

        if total_categories > config.MAX_CATEGORIES:
            self.log(f"[WARN] Warning: {total_categories} categories exceeds recommended max of {config.MAX_CATEGORIES}")

        self.log(f"[OK] Final output validated successfully")
        self.log(f"  - Total rows: {len(df)}")
        self.log(f"  - Total categories: {total_categories}")
        self.log(f"  - Largest category: {category_counts.iloc[0]} tickets ({category_counts.index[0]})")

    def get_validation_report(self) -> str:
        """Generate a validation report."""
        report = [
            "=" * 50,
            "VALIDATION REPORT",
            "=" * 50,
            f"Original row count: {self.original_row_count}",
            f"Original columns: {sorted(self.original_columns)}",
            "",
            "Validation Log:",
            "-" * 30,
        ]
        report.extend(self.validation_log)
        report.append("=" * 50)

        return "\n".join(report)


def create_backup(df: pd.DataFrame, original_path: str) -> str:
    """Create a backup of the original data before processing."""
    from pathlib import Path
    from datetime import datetime

    backup_dir = Path(original_path).parent / "backups"
    backup_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = backup_dir / f"{Path(original_path).stem}_backup_{timestamp}.xlsx"

    df.to_excel(backup_path, index=False)
    print(f"[OK] Backup created: {backup_path}")

    return str(backup_path)
