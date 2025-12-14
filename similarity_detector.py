"""Similarity detection to ensure consistent categorization across similar tickets."""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import Optional
import config


class SimilarityDetector:
    """Detects similar tickets to ensure consistent categorization."""

    def __init__(self, threshold: float = config.SIMILARITY_THRESHOLD):
        self.threshold = threshold
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 2),
            max_features=5000
        )
        self.ticket_texts: list[str] = []
        self.ticket_indices: list[int] = []
        self.tfidf_matrix = None

    def add_tickets(self, tickets: list[dict]) -> None:
        """Add tickets to the similarity index."""
        for ticket in tickets:
            # Combine short description, description, and solution for better matching
            text = f"{ticket['short_description']} {ticket['description']} {ticket['solution']}"
            self.ticket_texts.append(text)
            self.ticket_indices.append(ticket['index'])

    def build_index(self) -> None:
        """Build the TF-IDF index for similarity detection."""
        if not self.ticket_texts:
            return

        self.tfidf_matrix = self.vectorizer.fit_transform(self.ticket_texts)

    def find_similar_tickets(self, ticket_idx: int) -> list[tuple[int, float]]:
        """Find tickets similar to the given ticket.

        Args:
            ticket_idx: Index of the ticket to find similar tickets for

        Returns:
            List of (ticket_index, similarity_score) tuples
        """
        if self.tfidf_matrix is None:
            return []

        try:
            position = self.ticket_indices.index(ticket_idx)
        except ValueError:
            return []

        ticket_vector = self.tfidf_matrix[position]
        similarities = cosine_similarity(ticket_vector, self.tfidf_matrix).flatten()

        similar = []
        for i, score in enumerate(similarities):
            if i != position and score >= self.threshold:
                similar.append((self.ticket_indices[i], float(score)))

        return sorted(similar, key=lambda x: x[1], reverse=True)

    def get_similarity_groups(self) -> list[set[int]]:
        """Group similar tickets together.

        Returns:
            List of sets, where each set contains indices of similar tickets
        """
        if self.tfidf_matrix is None:
            self.build_index()

        if self.tfidf_matrix is None or self.tfidf_matrix.shape[0] == 0:
            return []

        # Compute pairwise similarities
        similarity_matrix = cosine_similarity(self.tfidf_matrix)

        # Build groups using union-find approach
        visited = set()
        groups = []

        for i in range(len(self.ticket_indices)):
            if i in visited:
                continue

            # Start a new group
            group = {self.ticket_indices[i]}
            queue = [i]
            visited.add(i)

            while queue:
                current = queue.pop(0)
                for j in range(len(self.ticket_indices)):
                    if j not in visited and similarity_matrix[current, j] >= self.threshold:
                        visited.add(j)
                        queue.append(j)
                        group.add(self.ticket_indices[j])

            if len(group) > 1:  # Only include groups with multiple tickets
                groups.append(group)

        return groups

    def enforce_category_consistency(
        self,
        categories: dict[int, str],
        tickets: list[dict]
    ) -> dict[int, str]:
        """Ensure similar tickets have the same category.

        Uses majority voting within similarity groups to determine the category.

        Args:
            categories: Current category assignments
            tickets: List of ticket dictionaries

        Returns:
            Updated categories with consistent assignments
        """
        self.add_tickets(tickets)
        self.build_index()

        groups = self.get_similarity_groups()
        updated_categories = categories.copy()

        for group in groups:
            # Get all categories in this group
            group_categories = {}
            for idx in group:
                cat = categories.get(idx)
                if cat:
                    group_categories[cat] = group_categories.get(cat, 0) + 1

            if not group_categories:
                continue

            # Find the most common category (majority vote)
            majority_category = max(group_categories.items(), key=lambda x: x[1])[0]

            # Update all tickets in the group to use the majority category
            changed = []
            for idx in group:
                if categories.get(idx) != majority_category:
                    changed.append((idx, categories.get(idx), majority_category))
                updated_categories[idx] = majority_category

            if changed:
                print(f"  Similarity group unified: {len(group)} tickets -> '{majority_category}'")
                for idx, old_cat, new_cat in changed[:3]:  # Show first 3 changes
                    print(f"    Ticket {idx}: '{old_cat}' -> '{new_cat}'")
                if len(changed) > 3:
                    print(f"    ... and {len(changed) - 3} more")

        return updated_categories

    def get_statistics(self) -> dict:
        """Get statistics about the similarity analysis."""
        groups = self.get_similarity_groups()
        return {
            "total_tickets": len(self.ticket_indices),
            "similarity_groups": len(groups),
            "tickets_in_groups": sum(len(g) for g in groups),
            "largest_group_size": max(len(g) for g in groups) if groups else 0,
        }
