"""Main ServiceNow Ticket Analysis Agent - Orchestrates the entire categorization process."""

import sys
from pathlib import Path
from typing import Optional
from tqdm import tqdm

from excel_handler import ExcelHandler, ExcelValidationError
from categorization_engine import CategorizationEngine, CategorizationError
from similarity_detector import SimilarityDetector
from validator import DataValidator, ValidationError, create_backup
import config


class SnowTicketAnalysisAgent:
    """
    ServiceNow Ticket Analysis Agent

    This agent analyzes ServiceNow tickets and intelligently categorizes them
    to help identify patterns and frequent issue types.

    Process:
    1. Read and validate Excel file
    2. Create backup of original data
    3. Batch process tickets through Claude for categorization
    4. Apply similarity detection to ensure consistency
    5. Validate all results
    6. Save categorized data with summary
    """

    def __init__(self, input_file: str, output_file: Optional[str] = None):
        self.input_file = input_file
        self.output_file = output_file
        self.excel_handler = ExcelHandler(input_file)
        self.categorization_engine = CategorizationEngine()
        self.similarity_detector = SimilarityDetector()
        self.validator: Optional[DataValidator] = None

    def run(self) -> Path:
        """Execute the full analysis pipeline.

        Returns:
            Path to the output file
        """
        print("\n" + "=" * 60)
        print("ServiceNow Ticket Analysis Agent")
        print("=" * 60)

        # Step 1: Read and validate input
        print("\n[1/6] Reading and validating input file...")
        df = self.excel_handler.read_excel()
        self.validator = DataValidator(df)

        # Step 2: Create backup
        print("\n[2/6] Creating backup...")
        create_backup(df, self.input_file)

        # Step 3: Extract ticket data
        print("\n[3/6] Extracting ticket data...")
        tickets = self.excel_handler.get_ticket_data()
        print(f"  Extracted {len(tickets)} tickets for analysis")

        # Step 4: Categorize tickets in batches
        print("\n[4/6] Categorizing tickets using Claude AI...")
        all_categories = self._categorize_all_tickets(tickets)

        # Step 5: Apply similarity-based consistency
        print("\n[5/6] Applying similarity-based consistency checks...")
        consistent_categories = self.similarity_detector.enforce_category_consistency(
            all_categories, tickets
        )

        # Normalize similar category names
        consistent_categories = self.categorization_engine.validate_and_normalize_categories(
            consistent_categories
        )

        # Step 6: Validate and save
        print("\n[6/6] Validating and saving results...")
        self.validator.validate_categories(consistent_categories)

        # Add categories to DataFrame
        self.excel_handler.add_categories(consistent_categories)

        # Final validation
        self.validator.validate_final_output(self.excel_handler.df)

        # Save output
        output_path = self.excel_handler.save_excel(self.output_file)

        # Generate and print summary
        self._print_summary()

        print("\n" + "=" * 60)
        print("Analysis Complete!")
        print("=" * 60)
        print(f"\nOutput file: {output_path}")

        return output_path

    def _categorize_all_tickets(self, tickets: list[dict]) -> dict[int, str]:
        """Categorize all tickets in batches."""
        all_categories = {}
        batch_size = config.BATCH_SIZE

        # Create batches
        batches = [
            tickets[i:i + batch_size]
            for i in range(0, len(tickets), batch_size)
        ]

        print(f"  Processing {len(batches)} batches of up to {batch_size} tickets each...")

        for batch_num, batch in enumerate(tqdm(batches, desc="  Categorizing"), 1):
            try:
                # Get existing categories for consistency
                existing_cats = self.categorization_engine.get_established_categories()

                # Categorize this batch
                batch_categories = self.categorization_engine.categorize_batch(
                    batch, existing_cats
                )

                all_categories.update(batch_categories)

            except CategorizationError as e:
                print(f"\n  ⚠ Error in batch {batch_num}: {e}")
                print("  Retrying batch...")

                # Retry once with smaller batches
                for ticket in batch:
                    if ticket['index'] not in all_categories:
                        try:
                            single_result = self.categorization_engine.categorize_batch(
                                [ticket],
                                self.categorization_engine.get_established_categories()
                            )
                            all_categories.update(single_result)
                        except CategorizationError as retry_error:
                            # Assign a default category for failed tickets
                            all_categories[ticket['index']] = "Uncategorized"
                            print(f"    Failed to categorize ticket {ticket['index']}: {retry_error}")

        # Verify all tickets are categorized
        missing = set(t['index'] for t in tickets) - set(all_categories.keys())
        if missing:
            raise CategorizationError(f"Failed to categorize tickets: {missing}")

        unique_categories = set(all_categories.values())
        print(f"\n  [OK] Categorization complete: {len(unique_categories)} categories identified")

        return all_categories

    def _print_summary(self) -> None:
        """Print a summary of the categorization results."""
        print("\n" + "-" * 40)
        print("CATEGORY SUMMARY")
        print("-" * 40)

        summary = self.excel_handler.generate_summary()
        print(summary.to_string())

        print("\n" + "-" * 40)

        # Similarity statistics
        stats = self.similarity_detector.get_statistics()
        if stats['similarity_groups'] > 0:
            print("\nSIMILARITY ANALYSIS:")
            print(f"  Total tickets: {stats['total_tickets']}")
            print(f"  Similarity groups found: {stats['similarity_groups']}")
            print(f"  Tickets in groups: {stats['tickets_in_groups']}")
            print(f"  Largest group size: {stats['largest_group_size']}")


def main():
    """Main entry point for the agent."""
    import argparse

    parser = argparse.ArgumentParser(
        description="ServiceNow Ticket Analysis Agent - Intelligently categorize support tickets"
    )
    parser.add_argument(
        "input_file",
        help="Path to the input Excel file (.xlsx or .xls)"
    )
    parser.add_argument(
        "-o", "--output",
        help="Path for output file (default: input_file_categorized.xlsx)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=config.BATCH_SIZE,
        help=f"Number of tickets per API batch (default: {config.BATCH_SIZE})"
    )

    args = parser.parse_args()

    # Update config if batch size specified
    if args.batch_size:
        config.BATCH_SIZE = args.batch_size

    try:
        agent = SnowTicketAnalysisAgent(args.input_file, args.output)
        output_path = agent.run()
        sys.exit(0)

    except (ExcelValidationError, CategorizationError, ValidationError) as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)

    except KeyboardInterrupt:
        print("\n\n⚠ Operation cancelled by user")
        sys.exit(130)

    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
