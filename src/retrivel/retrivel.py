import os
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct
from typing import List, Any
import pandas as pd


class Retrivel:
    def __init__(self):
        self.client = QdrantClient(
            url=os.getenv("QDRANT_URL"),
        )

    def get_points(self, collection_name: str) -> list[PointStruct]:
        """
        Retrieve documents of a specific type.
        """
        if collection_name not in ["fornitori", "bandi"]:
            raise ValueError("Invalid type provided. Must be 'fornitori' or 'bandi'.")
        # Retrieve all points from the specified collection
        points, _ = self.client.scroll(
            collection_name=collection_name,
            with_payload=True,
            with_vectors=True,
        )
        return points

    def query_and_collect(self, limit: int = 5) -> List[List[Any]]:
        """
        Fetch all points from 'fornitori', then for each point query 'bandi' across three
        modalities ('dense', 'sparse', 'colbertv2.0'), normalize the scores, and return
        a list of [fornitore_name, bando_name, normalized_score].
        """
        # Fetch all 'fornitori' points

        data: List[List[Any]] = []

        for point in self.get_points("fornitori"):
            # Query 'bandi' with prefetch modalities
            resp = self.client.query_points(
                collection_name="bandi",
                prefetch=[
                    {
                        "query": point.vector.get("dense"),
                        "using": "dense",
                        "limit": limit,
                    },
                    {
                        "query": point.vector.get("sparse"),
                        "using": "sparse",
                        "limit": limit,
                    },
                ],
                query=point.vector.get("colbertv2.0"),
                using="colbertv2.0",
                with_payload=True,
                with_vectors=True,
                limit=limit,
            )

            # Source metadata
            fornitore_name = point.payload.get("metadata", {}).get("name", "unknown")
            colbert_vec = point.vector.get("colbertv2.0") or []
            query_length = len(colbert_vec) or 1

            # Collect and normalize scores
            for p in resp.points:
                raw_score = p.score
                normalized = raw_score / query_length
                normalized_score = max(0.0, min(1.0, normalized))

                bando_name = p.payload.get("metadata", {}).get("name", "unknown")
                data.append([fornitore_name, bando_name, normalized_score])

        return data

    def get_scores_matrix(self, limit: int = 5) -> List[List[float]]:
        """
        Generates a matrix of normalized scores for suppliers ("Fornitori") and tenders ("Bandi").

        This method queries and collects data, constructs a DataFrame, and pivots it to create a matrix
        where each row represents a supplier, each column represents a tender, and each cell contains
        the normalized score for that supplier-tender pair. Missing values are filled with zero.

        Args:
            limit (int, optional): The maximum number of records to retrieve. Defaults to 5.

        Returns:
            List[List[float]]: A 2D list (matrix) of normalized scores, with rows as suppliers and columns as tenders.
        """
        scores_df = pd.DataFrame(
            self.query_and_collect(limit),
            columns=["Fornitori", "Bandi", "Normalized Score"],
        )

        scores_matrix = scores_df.pivot(
            index="Fornitori", columns="Bandi", values="Normalized Score"
        ).fillna(0)

        return scores_matrix
