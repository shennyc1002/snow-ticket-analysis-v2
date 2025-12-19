"""Text Preprocessor - Cleans and truncates ticket text before sending to LLM."""

import re
from typing import Optional


class TextPreprocessor:
    """Cleans email trails, signatures, and truncates long text intelligently."""

    def __init__(self, max_length: int = 1000, min_length: int = 50):
        """
        Args:
            max_length: Maximum characters to keep per field
            min_length: Minimum characters needed for meaningful categorization
        """
        self.max_length = max_length
        self.min_length = min_length

        # Patterns to remove
        self.removal_patterns = [
            # Email confidentiality notices
            re.compile(
                r'(?:CONFIDENTIAL(?:ITY)?|DISCLAIMER|NOTICE|PRIVILEGED)[\s:]*.*?(?=\n\n|\Z)',
                re.IGNORECASE | re.DOTALL
            ),
            re.compile(
                r'This (?:e-?mail|message|communication) (?:and any|is|contains).*?(?:intended recipient|unauthorized|prohibited|confidential).*?(?=\n\n|\Z)',
                re.IGNORECASE | re.DOTALL
            ),
            re.compile(
                r'If you (?:are not|have received).*?(?:intended recipient|delete|notify|error).*?(?=\n\n|\Z)',
                re.IGNORECASE | re.DOTALL
            ),

            # Email signatures
            re.compile(
                r'(?:^|\n)[-_]{2,}\s*\n.*?(?=\n\n|\Z)',  # Lines starting with -- or __
                re.DOTALL
            ),
            re.compile(
                r'(?:Best regards|Kind regards|Regards|Thanks|Thank you|Sincerely|Cheers),?\s*\n.*?(?=\n\n|\Z)',
                re.IGNORECASE | re.DOTALL
            ),
            re.compile(
                r'(?:Sent from my (?:iPhone|Android|iPad|Mobile)).*?(?=\n|\Z)',
                re.IGNORECASE
            ),

            # Email headers in trails
            re.compile(
                r'(?:^|\n)(?:From|To|Cc|Bcc|Subject|Date|Sent):\s*.*?(?=\n)',
                re.IGNORECASE | re.MULTILINE
            ),
            re.compile(
                r'(?:^|\n)[-]+\s*(?:Original Message|Forwarded (?:Message|message)|Reply).*?[-]+',
                re.IGNORECASE
            ),
            re.compile(
                r'On\s+(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun).*?wrote:',
                re.IGNORECASE | re.DOTALL
            ),

            # Common footer patterns
            re.compile(
                r'(?:This email was sent|Unsubscribe|Click here to).*?(?=\n\n|\Z)',
                re.IGNORECASE | re.DOTALL
            ),

            # Multiple blank lines -> single blank line
            re.compile(r'\n{3,}'),

            # Excessive whitespace
            re.compile(r'[ \t]{3,}'),
        ]

        # Email trail separators - keep only first message
        self.email_trail_patterns = [
            re.compile(r'\n[-_=]{3,}\s*(?:Original|Forwarded).*', re.IGNORECASE | re.DOTALL),
            re.compile(r'\nFrom:.*?(?:wrote|sent):', re.IGNORECASE | re.DOTALL),
            re.compile(r'\n>.*(?:\n>.*)*', re.DOTALL),  # Quoted text with >
        ]

    def clean_text(self, text: Optional[str]) -> str:
        """
        Remove email signatures, confidentiality notices, and clean up text.

        Args:
            text: Raw text that may contain email trails

        Returns:
            Cleaned text
        """
        if not text:
            return ""

        result = str(text)

        # First, try to extract only the first email in a trail
        for pattern in self.email_trail_patterns:
            match = pattern.search(result)
            if match:
                result = result[:match.start()]

        # Remove noise patterns
        for pattern in self.removal_patterns:
            result = pattern.sub('\n', result)

        # Clean up whitespace
        result = re.sub(r'\n{2,}', '\n\n', result)  # Max 2 newlines
        result = re.sub(r'[ \t]+', ' ', result)  # Single spaces
        result = result.strip()

        return result

    def truncate_smart(self, text: str, max_length: Optional[int] = None) -> str:
        """
        Truncate text intelligently, keeping the most important parts.

        Strategy:
        - Keep the first part (usually contains the main issue)
        - If text is too long, add "..." and keep beginning

        Args:
            text: Text to truncate
            max_length: Override default max_length

        Returns:
            Truncated text
        """
        max_len = max_length or self.max_length

        if len(text) <= max_len:
            return text

        # Keep first portion (usually contains the main issue)
        # Try to break at sentence or paragraph boundary
        truncated = text[:max_len]

        # Try to end at a sentence
        last_period = truncated.rfind('.')
        last_newline = truncated.rfind('\n')

        # Use the later of period or newline, if reasonable
        break_point = max(last_period, last_newline)
        if break_point > max_len * 0.7:  # At least 70% of max length
            truncated = truncated[:break_point + 1]

        return truncated.strip() + " [TRUNCATED]"

    def process(self, text: Optional[str], max_length: Optional[int] = None) -> str:
        """
        Full preprocessing: clean and truncate.

        Args:
            text: Raw text
            max_length: Optional max length override

        Returns:
            Cleaned and truncated text
        """
        cleaned = self.clean_text(text)
        return self.truncate_smart(cleaned, max_length)

    def process_ticket(self, ticket: dict, max_description_length: int = 1000,
                       max_solution_length: int = 500) -> dict:
        """
        Process a ticket dictionary.

        Args:
            ticket: Ticket dict with description, solution fields
            max_description_length: Max chars for description
            max_solution_length: Max chars for solution

        Returns:
            Processed ticket dict
        """
        processed = ticket.copy()

        # Process description (longer, usually has more detail)
        if 'description' in processed and processed['description']:
            processed['description'] = self.process(
                processed['description'],
                max_length=max_description_length
            )

        # Process solution (usually shorter)
        if 'solution' in processed and processed['solution']:
            processed['solution'] = self.process(
                processed['solution'],
                max_length=max_solution_length
            )

        # Short description usually doesn't need truncation, just clean
        if 'short_description' in processed and processed['short_description']:
            processed['short_description'] = self.clean_text(
                processed['short_description']
            )[:200]  # Hard limit 200 chars

        return processed

    def has_sufficient_context(self, ticket: dict) -> bool:
        """
        Check if ticket has enough information for meaningful categorization.

        Args:
            ticket: Processed ticket dict

        Returns:
            True if sufficient context exists
        """
        short_desc = str(ticket.get('short_description', '')).strip()
        description = str(ticket.get('description', '')).strip()
        solution = str(ticket.get('solution', '')).strip()

        # Combined text length
        total_text = len(short_desc) + len(description) + len(solution)

        # Must have at least short_description OR some description
        has_short_desc = len(short_desc) >= 10
        has_description = len(description) >= self.min_length
        has_solution = len(solution) >= 20

        return has_short_desc or has_description or has_solution

    def process_tickets(self, tickets: list[dict],
                        max_description_length: int = 1000,
                        max_solution_length: int = 500) -> list[dict]:
        """
        Process multiple tickets.

        Args:
            tickets: List of ticket dicts
            max_description_length: Max chars for description
            max_solution_length: Max chars for solution

        Returns:
            List of processed tickets with low-context warnings
        """
        processed = []

        for ticket in tickets:
            proc_ticket = self.process_ticket(
                ticket,
                max_description_length=max_description_length,
                max_solution_length=max_solution_length
            )

            # Add warning flag if insufficient context
            if not self.has_sufficient_context(proc_ticket):
                proc_ticket['_low_context'] = True

            processed.append(proc_ticket)

        return processed


# Singleton instance
_preprocessor = TextPreprocessor()


def process_text(text: str, max_length: int = 1000) -> str:
    """Convenience function to clean and truncate text."""
    return _preprocessor.process(text, max_length)


def process_ticket(ticket: dict) -> dict:
    """Convenience function to process a single ticket."""
    return _preprocessor.process_ticket(ticket)


def process_tickets(tickets: list[dict]) -> list[dict]:
    """Convenience function to process multiple tickets."""
    return _preprocessor.process_tickets(tickets)
