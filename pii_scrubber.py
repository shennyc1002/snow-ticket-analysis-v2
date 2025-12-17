"""PII Scrubber - Removes sensitive information before sending to LLM."""

import re
from typing import Optional


class PIIScrubber:
    """Removes PII (Personally Identifiable Information) from text."""

    def __init__(self):
        # Compile regex patterns for performance
        self.patterns = {
            # Email addresses
            "email": re.compile(
                r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            ),

            # Phone numbers (various formats)
            "phone": re.compile(
                r'''
                (?:
                    (?:\+?1[-.\s]?)?              # Optional country code
                    (?:\(?\d{3}\)?[-.\s]?)        # Area code
                    \d{3}[-.\s]?\d{4}             # Main number
                )
                |
                (?:
                    \+\d{1,3}[-.\s]?\d{1,4}[-.\s]?\d{1,4}[-.\s]?\d{1,9}  # International
                )
                ''',
                re.VERBOSE
            ),

            # Social Security Numbers (US)
            "ssn": re.compile(
                r'\b\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b'
            ),

            # Credit Card Numbers (basic patterns)
            "credit_card": re.compile(
                r'\b(?:\d{4}[-.\s]?){3}\d{4}\b|\b\d{15,16}\b'
            ),

            # IP Addresses
            "ip_address": re.compile(
                r'\b(?:\d{1,3}\.){3}\d{1,3}\b'
            ),

            # Names after common prefixes (Mr., Mrs., Ms., Dr., etc.)
            "name_prefix": re.compile(
                r'\b(?:Mr\.?|Mrs\.?|Ms\.?|Dr\.?|Prof\.?)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?',
                re.IGNORECASE
            ),

            # Employee IDs (common patterns like EMP001, E12345)
            "employee_id": re.compile(
                r'\b(?:EMP|E|ID)[-_]?\d{3,8}\b',
                re.IGNORECASE
            ),

            # URLs (may contain sensitive info)
            "url": re.compile(
                r'https?://[^\s<>"{}|\\^`\[\]]+',
                re.IGNORECASE
            ),

            # Windows file paths with usernames
            "windows_path": re.compile(
                r'[A-Za-z]:\\(?:Users|Documents and Settings)\\[^\\]+',
                re.IGNORECASE
            ),

            # Unix home paths
            "unix_path": re.compile(
                r'/(?:home|Users)/[a-zA-Z0-9_.-]+',
                re.IGNORECASE
            ),

            # Order IDs (common patterns like ORD-12345, ORDER-123456, #12345)
            "order_id": re.compile(
                r'\b(?:ORD|ORDER|PO|SO)[-_#]?\d{4,12}\b|#\d{5,12}\b',
                re.IGNORECASE
            ),

            # Batch/Job IDs (JOB-123, BATCH_456, job_id: 789)
            "batch_job_id": re.compile(
                r'\b(?:JOB|BATCH|TASK|PROCESS)[-_#:]?\s*\d{3,12}\b',
                re.IGNORECASE
            ),

            # Transaction IDs (TXN-123, TRANS_456)
            "transaction_id": re.compile(
                r'\b(?:TXN|TRANS|TRANSACTION)[-_#:]?\s*[A-Z0-9]{6,20}\b',
                re.IGNORECASE
            ),

            # Customer IDs (CUST-123, CID_456)
            "customer_id": re.compile(
                r'\b(?:CUST|CID|CUSTOMER)[-_#:]?\s*\d{4,12}\b',
                re.IGNORECASE
            ),

            # Account numbers
            "account_number": re.compile(
                r'\b(?:ACCT|ACC|ACCOUNT)[-_#:]?\s*\d{6,15}\b',
                re.IGNORECASE
            ),

            # Case/Ticket numbers (INC123456, RITM0012345, REQ000123)
            "ticket_number": re.compile(
                r'\b(?:INC|RITM|REQ|CHG|PRB|TASK)\d{6,12}\b',
                re.IGNORECASE
            ),

            # Server/Host names (server01.domain.com, HOST-PROD-01)
            "hostname": re.compile(
                r'\b(?:[a-zA-Z]+-)?(?:PROD|DEV|UAT|QA|STG|TEST)[-_]?[a-zA-Z0-9]+(?:\.[a-zA-Z0-9.-]+)?\b',
                re.IGNORECASE
            ),

            # Database names (DB_PROD, database: mydb)
            "database": re.compile(
                r'\b(?:DB|DATABASE)[-_:]?\s*[A-Z0-9_]{3,30}\b',
                re.IGNORECASE
            ),
        }

        # Replacement tokens
        self.replacements = {
            "email": "[EMAIL]",
            "phone": "[PHONE]",
            "ssn": "[SSN]",
            "credit_card": "[CREDIT_CARD]",
            "ip_address": "[IP_ADDRESS]",
            "name_prefix": "[NAME]",
            "employee_id": "[EMPLOYEE_ID]",
            "url": "[URL]",
            "windows_path": "[FILE_PATH]",
            "unix_path": "[FILE_PATH]",
            "order_id": "[ORDER_ID]",
            "batch_job_id": "[JOB_ID]",
            "transaction_id": "[TRANSACTION_ID]",
            "customer_id": "[CUSTOMER_ID]",
            "account_number": "[ACCOUNT_NUMBER]",
            "ticket_number": "[TICKET_NUMBER]",
            "hostname": "[HOSTNAME]",
            "database": "[DATABASE]",
        }

    def scrub(self, text: Optional[str]) -> str:
        """
        Remove PII from text.

        Args:
            text: Input text that may contain PII

        Returns:
            Text with PII replaced by tokens
        """
        if not text:
            return text or ""

        result = text

        for pii_type, pattern in self.patterns.items():
            replacement = self.replacements.get(pii_type, "[REDACTED]")
            result = pattern.sub(replacement, result)

        return result

    def scrub_ticket(self, ticket: dict) -> dict:
        """
        Scrub PII from a ticket dictionary.

        Args:
            ticket: Ticket dict with fields like short_description, description, solution

        Returns:
            New ticket dict with PII removed
        """
        scrubbed = ticket.copy()

        # Fields to scrub
        text_fields = ['short_description', 'description', 'solution']

        for field in text_fields:
            if field in scrubbed and scrubbed[field]:
                scrubbed[field] = self.scrub(str(scrubbed[field]))

        return scrubbed

    def scrub_tickets(self, tickets: list[dict]) -> list[dict]:
        """
        Scrub PII from a list of tickets.

        Args:
            tickets: List of ticket dictionaries

        Returns:
            List of tickets with PII removed
        """
        return [self.scrub_ticket(ticket) for ticket in tickets]


# Singleton instance for easy import
_scrubber = PIIScrubber()


def scrub_text(text: str) -> str:
    """Convenience function to scrub text."""
    return _scrubber.scrub(text)


def scrub_ticket(ticket: dict) -> dict:
    """Convenience function to scrub a ticket."""
    return _scrubber.scrub_ticket(ticket)


def scrub_tickets(tickets: list[dict]) -> list[dict]:
    """Convenience function to scrub multiple tickets."""
    return _scrubber.scrub_tickets(tickets)
