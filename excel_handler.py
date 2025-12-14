"""Excel file handling with validation for ServiceNow ticket data."""

import pandas as pd
from pathlib import Path
from typing import Optional
import config


class ExcelValidationError(Exception):
    """Raised when Excel validation fails."""
    pass


class ExcelHandler:
    """Handles reading and writing Excel files with validation."""

    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
        self.df: Optional[pd.DataFrame] = None
        self.original_row_count: int = 0

    def validate_file_exists(self) -> None:
        """Ensure the file exists and is readable."""
        if not self.file_path.exists():
            raise ExcelValidationError(f"File not found: {self.file_path}")
        if not self.file_path.suffix.lower() in ['.xlsx', '.xls']:
            raise ExcelValidationError(f"Invalid file type. Expected .xlsx or .xls, got: {self.file_path.suffix}")

    def validate_columns(self, df: pd.DataFrame) -> None:
        """Validate required columns exist in the DataFrame."""
        required_columns = [
            config.COLUMN_MAPPING["short_description"],
            config.COLUMN_MAPPING["description"],
            config.COLUMN_MAPPING["solution"],
            config.COLUMN_MAPPING["priority"],
        ]

        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ExcelValidationError(
                f"Missing required columns: {missing_columns}. "
                f"Available columns: {df.columns.tolist()}"
            )

    def read_excel(self) -> pd.DataFrame:
        """Read and validate the Excel file."""
        self.validate_file_exists()

        try:
            self.df = pd.read_excel(self.file_path)
        except Exception as e:
            raise ExcelValidationError(f"Failed to read Excel file: {e}")

        self.validate_columns(self.df)
        self.original_row_count = len(self.df)

        if self.original_row_count == 0:
            raise ExcelValidationError("Excel file contains no data rows")

        print(f"[OK] Successfully loaded {self.original_row_count} tickets from {self.file_path.name}")
        print(f"  Columns: {self.df.columns.tolist()}")

        return self.df

    def get_ticket_data(self) -> list[dict]:
        """Extract ticket data for categorization."""
        if self.df is None:
            raise ExcelValidationError("No data loaded. Call read_excel() first.")

        tickets = []
        for idx, row in self.df.iterrows():
            ticket = {
                "index": idx,
                "incident_number": row.get(config.COLUMN_MAPPING["incident_number"], f"ROW_{idx}"),
                "short_description": str(row.get(config.COLUMN_MAPPING["short_description"], "")),
                "description": str(row.get(config.COLUMN_MAPPING["description"], "")),
                "solution": str(row.get(config.COLUMN_MAPPING["solution"], "")),
                "priority": str(row.get(config.COLUMN_MAPPING["priority"], "")),
            }
            tickets.append(ticket)

        return tickets

    def add_categories(self, categories: dict[int, str]) -> pd.DataFrame:
        """Add category column to DataFrame.

        Args:
            categories: Dictionary mapping row index to category name

        Returns:
            Updated DataFrame with Category column
        """
        if self.df is None:
            raise ExcelValidationError("No data loaded. Call read_excel() first.")

        # Validate we have categories for all rows
        missing_indices = set(range(len(self.df))) - set(categories.keys())
        if missing_indices:
            raise ExcelValidationError(
                f"Missing categories for {len(missing_indices)} rows: {list(missing_indices)[:10]}..."
            )

        # Add category column
        self.df[config.COLUMN_MAPPING["category"]] = self.df.index.map(categories)

        # Validate no null categories
        null_categories = self.df[self.df[config.COLUMN_MAPPING["category"]].isna()]
        if len(null_categories) > 0:
            raise ExcelValidationError(
                f"Found {len(null_categories)} rows with null categories"
            )

        return self.df

    def save_excel(self, output_path: Optional[str] = None) -> Path:
        """Save the categorized DataFrame to Excel.

        Args:
            output_path: Optional custom output path. If None, uses input filename with suffix.

        Returns:
            Path to the saved file
        """
        if self.df is None:
            raise ExcelValidationError("No data to save. Process the file first.")

        if output_path:
            output_file = Path(output_path)
        else:
            output_file = self.file_path.parent / f"{self.file_path.stem}{config.OUTPUT_SUFFIX}.xlsx"

        # Final validation before saving
        if len(self.df) != self.original_row_count:
            raise ExcelValidationError(
                f"Row count mismatch! Original: {self.original_row_count}, Current: {len(self.df)}. "
                "Aborting to prevent data loss."
            )

        if config.COLUMN_MAPPING["category"] not in self.df.columns:
            raise ExcelValidationError("Category column not found. Run categorization first.")

        # Save with formatting
        self.df.to_excel(output_file, index=False, engine='openpyxl')

        print(f"[OK] Saved categorized data to: {output_file}")
        print(f"  Total rows: {len(self.df)}")
        print(f"  Categories: {self.df[config.COLUMN_MAPPING['category']].nunique()} unique categories")

        return output_file

    def generate_summary(self) -> pd.DataFrame:
        """Generate a category frequency summary."""
        if self.df is None or config.COLUMN_MAPPING["category"] not in self.df.columns:
            raise ExcelValidationError("No categorized data available")

        summary = self.df.groupby(config.COLUMN_MAPPING["category"]).agg({
            config.COLUMN_MAPPING["incident_number"]: 'count',
            config.COLUMN_MAPPING["priority"]: lambda x: x.mode().iloc[0] if len(x) > 0 else 'N/A'
        }).rename(columns={
            config.COLUMN_MAPPING["incident_number"]: 'Ticket Count',
            config.COLUMN_MAPPING["priority"]: 'Most Common Priority'
        }).sort_values('Ticket Count', ascending=False)

        return summary
