"""Intelligent categorization engine using GPT API (wrapper or direct)."""

import json
from typing import Optional
from gpt_client import get_gpt_client, GPTClientError
import config


class CategorizationError(Exception):
    """Raised when categorization fails."""
    pass


class CategorizationEngine:
    """Uses GPT to intelligently categorize ServiceNow tickets."""

    def __init__(self):
        try:
            self.client = get_gpt_client()
        except GPTClientError as e:
            raise CategorizationError(str(e))
        self.established_categories: list[str] = []

    def _build_categorization_prompt(self, tickets: list[dict], existing_categories: list[str]) -> str:
        """Build the prompt for categorizing tickets."""
        tickets_text = "\n\n".join([
            f"TICKET {t['index']}:\n"
            f"  Incident: {t['incident_number']}\n"
            f"  Short Description: {t['short_description']}\n"
            f"  Description: {t['description']}\n"
            f"  Solution: {t['solution']}\n"
            f"  Priority: {t['priority']}"
            for t in tickets
        ])

        existing_cats_text = ""
        if existing_categories:
            existing_cats_text = f"""
IMPORTANT - Previously established categories (REUSE these when applicable):
{json.dumps(existing_categories, indent=2)}

You MUST use an existing category if the ticket fits. Only create a new category if none of the existing ones are appropriate.
"""

        return f"""You are a ServiceNow ticket categorization expert. Your task is to analyze IT support tickets and assign them to appropriate categories.

RULES:
1. Categories should be clear, concise, and business-meaningful (e.g., "Order Management", "Password Reset", "System Access", "Data Correction")
2. Categories should be broad enough to group similar issues but specific enough to be useful
3. Use consistent naming conventions (Title Case, no special characters)
4. Similar issues MUST get the same category - this is critical for accurate reporting
5. Maximum {config.MAX_CATEGORIES} total categories allowed
{existing_cats_text}

TICKETS TO CATEGORIZE:
{tickets_text}

Respond with a JSON object mapping ticket index to category. Example format:
{{"0": "Order Management", "1": "Password Reset", "2": "Order Management"}}

IMPORTANT:
- Return ONLY the JSON object, no additional text
- Every ticket index must have a category assigned
- Categories must be meaningful and descriptive"""

    def categorize_batch(self, tickets: list[dict], existing_categories: Optional[list[str]] = None) -> dict[int, str]:
        """Categorize a batch of tickets.

        Args:
            tickets: List of ticket dictionaries
            existing_categories: Previously established categories to maintain consistency

        Returns:
            Dictionary mapping ticket index to category
        """
        if not tickets:
            return {}

        if existing_categories is None:
            existing_categories = self.established_categories

        prompt = self._build_categorization_prompt(tickets, existing_categories)

        try:
            response_text = self.client.get_completion_text(
                messages=[{"role": "user", "content": prompt}],
                model=config.MODEL_NAME,
                max_tokens=config.MAX_TOKENS,
                response_format={"type": "json_object"}
            ).strip()

            # Parse JSON response
            # Handle potential markdown code blocks (fallback)
            if response_text.startswith("```"):
                response_text = response_text.split("```")[1]
                if response_text.startswith("json"):
                    response_text = response_text[4:]
                response_text = response_text.strip()

            categories = json.loads(response_text)

            # Convert string keys to integers and validate
            result = {}
            for key, value in categories.items():
                idx = int(key)
                if not isinstance(value, str) or not value.strip():
                    raise CategorizationError(f"Invalid category for ticket {idx}: {value}")
                result[idx] = value.strip()

                # Track new categories
                if value.strip() not in self.established_categories:
                    self.established_categories.append(value.strip())

            # Validate all tickets got categories
            expected_indices = {t['index'] for t in tickets}
            missing = expected_indices - set(result.keys())
            if missing:
                raise CategorizationError(f"Missing categories for tickets: {missing}")

            return result

        except json.JSONDecodeError as e:
            raise CategorizationError(f"Failed to parse API response as JSON: {e}\nResponse: {response_text}")
        except GPTClientError as e:
            raise CategorizationError(f"GPT API error: {e}")
        except Exception as e:
            raise CategorizationError(f"Categorization failed: {e}")

    def validate_and_normalize_categories(self, all_categories: dict[int, str]) -> dict[int, str]:
        """Second pass: Validate and normalize all categories for consistency.

        This ensures similar tickets that were processed in different batches
        end up with the same category.
        """
        # Build reverse mapping: category -> tickets
        category_tickets: dict[str, list[int]] = {}
        for idx, cat in all_categories.items():
            if cat not in category_tickets:
                category_tickets[cat] = []
            category_tickets[cat].append(idx)

        # Check for very similar category names and merge them
        normalized = all_categories.copy()
        categories_list = list(category_tickets.keys())

        # Simple similarity check - can be enhanced with fuzzy matching
        for i, cat1 in enumerate(categories_list):
            for cat2 in categories_list[i + 1:]:
                if self._are_similar_categories(cat1, cat2):
                    # Merge to the first one (or the more common one)
                    if len(category_tickets[cat1]) >= len(category_tickets[cat2]):
                        target, source = cat1, cat2
                    else:
                        target, source = cat2, cat1

                    for idx in category_tickets[source]:
                        normalized[idx] = target
                    print(f"  Merged similar categories: '{source}' -> '{target}'")

        return normalized

    def _are_similar_categories(self, cat1: str, cat2: str) -> bool:
        """Check if two category names are similar enough to merge."""
        c1 = cat1.lower().replace(" ", "").replace("-", "").replace("_", "")
        c2 = cat2.lower().replace(" ", "").replace("-", "").replace("_", "")

        # Exact match after normalization
        if c1 == c2:
            return True

        # One contains the other
        if c1 in c2 or c2 in c1:
            return True

        return False

    def get_established_categories(self) -> list[str]:
        """Return the list of established categories."""
        return self.established_categories.copy()
