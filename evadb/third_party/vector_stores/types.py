# coding=utf-8
# Copyright 2018-2023 EvaDB
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from dataclasses import dataclass
from typing import List, Optional, Dict


@dataclass
class FeaturePayload:
    id: int
    embedding: List[float]


@dataclass
class VectorIndexQuery:
    embedding: List[float]
    top_k: int


@dataclass
class VectorIndexQueryResult:
    similarities: List[float]
    ids: List[int]
    others: List[Dict]


@dataclass
class ObjectUpdatePayload:
    """Payload for updating an object."""
    uuid: str
    properties: Dict


@dataclass
class ObjectReplacePayload:
    """Payload for replacing an object."""
    uuid: str
    properties: Dict


@dataclass
class ObjectPropertyDeletionPayload:
    """Payload for deleting specific properties from an object."""
    uuid: str
    properties_to_delete: List[str]


@dataclass
class SearchQueryPayload:
    """Payload for performing a search query."""
    query: str
    properties: Optional[List[str]] = None
    limit: int = 10
    autocut: Optional[int] = None
    additional: Optional[List[str]] = None


@dataclass
class GenerativeSearchPayload:
    """Payload for performing a generative search query."""
    query: str
    prompt: str
    properties: Optional[List[str]] = None
    limit: int = 10
    grouped_task: bool = False
    grouped_properties: Optional[List[str]] = None


@dataclass
class LabelBasedSearchPayload:
    """Payload for performing a label-based search with filters."""
    properties: List[str]
    filters: Dict
    limit: int = 10

class VectorStore:
    def create(self, vector_dim: int):
        """Create an index"""
        ...

    def add(self, payload: List[FeaturePayload]) -> None:
        """Add embeddings to the vector store"""
        ...

    def persist(self) -> None:
        """Persist index to disk"""
        return None

    def query(self, query: VectorIndexQuery) -> VectorIndexQueryResult:
        """Query index"""
        ...

    def delete(self):
        """delete an index"""
        ...

    def update_object_properties(self, payload: ObjectUpdatePayload) -> None:
        """
        Update specific properties of an object in Weaviate.

        Args:
            payload (ObjectUpdatePayload): A dataclass containing the following fields:
                - uuid (str): The UUID of the object to update.
                - properties (Dict): A dictionary of properties to update, where each key-value pair represents a property name and its new value.

        Returns:
            None: This method does not return anything but updates the specified object properties in Weaviate.
        """
        ...

    def replace_object(self, payload: ObjectReplacePayload) -> None:
        """
        Replace an object in its entirety in Weaviate.

        Args:
            payload (ObjectReplacePayload): A dataclass containing the following fields:
                - uuid (str): The UUID of the object to replace.
                - properties (Dict): A dictionary representing the new properties of the object. This will replace the current properties entirely.

        Returns:
            None: This method does not return anything but replaces the specified object in Weaviate.
        """
        ...

    def delete_object_properties(self, payload: ObjectPropertyDeletionPayload) -> None:
        """
        Delete specific properties from an object in Weaviate.

        Args:
            payload (ObjectPropertyDeletionPayload): A dataclass containing the following fields:
                - uuid (str): The UUID of the object from which properties should be deleted.
                - properties_to_delete (List[str]): A list of property names to delete from the object.

        Returns:
            None: This method does not return anything but removes the specified properties from the object in Weaviate.
        """
        ...

    def keyword_search(self, payload: SearchQueryPayload) -> VectorIndexQueryResult:
        """
        Perform a keyword search in Weaviate.

        Args:
            payload (SearchQueryPayload): A dataclass containing the following fields:
                - query (str): The keyword query string for the search.
                - properties (List[str], optional): The properties to consider in the search.
                - limit (int, optional): The maximum number of results to return. Defaults to 10.
                - autocut (int, optional): A threshold to group results based on their score. If None, all results are returned.

        Returns:
            List[Dict]: A list of dictionaries containing the search results, with each dictionary representing a matching object from Weaviate.
        """
        raise NotImplementedError("Keyword search is not implemented for this vector store.")

    def hybrid_search(self, payload: SearchQueryPayload) -> VectorIndexQueryResult:
        """
        Perform a hybrid search in Weaviate, combining keyword and vector search.

        Args:
            payload (SearchQueryPayload): A dataclass containing the following fields:
                - query (str): The keyword query string for the BM25 search.
                - properties (List[str], optional): The properties to consider in the BM25 search.
                - limit (int, optional): The maximum number of results to return. Defaults to 10.
                - additional (List[str], optional): Additional properties to retrieve in the search results.

        Returns:
            List[Dict]: A list of dictionaries containing the hybrid search results, combining the relevance from both keyword and vector search.
        """
        raise NotImplementedError("Keyword search is not implemented for this vector store.")

    def generative_search(self, payload: GenerativeSearchPayload) -> VectorIndexQueryResult:
        """
        Perform a generative search in Weaviate using a large language model (LLM).

        Args:
            payload (GenerativeSearchPayload): A dataclass containing the following fields:
                - query (str): The keyword query string for initial object retrieval.
                - prompt (str): The prompt to use for generating text with the LLM.
                - properties (List[str], optional): The properties to consider in the initial search.
                - limit (int, optional): The maximum number of results to return. Defaults to 10.
                - grouped_task (bool, optional): Whether to perform a grouped task, using the prompt to generate grouped responses.
                - grouped_properties (List[str], optional): Properties to include in the prompt for a grouped task.

        Returns:
            List[Dict]: A list of dictionaries containing the search and generated results, with each dictionary representing an object and its generated content.
        """
        raise NotImplementedError("Keyword search is not implemented for this vector store.")

    def label_based_search(self, payload: LabelBasedSearchPayload) -> VectorIndexQueryResult:
        """
        Perform a label-based search in Weaviate with filters.

        Args:
            payload (LabelBasedSearchPayload): A dataclass containing the following fields:
                - properties (List[str]): The properties to retrieve from the search results.
                - filters (Dict): A dictionary defining the filters to apply, where each filter specifies a path, operator, and value to filter the search results.
                - limit (int, optional): The maximum number of results to return. Defaults to 10.

        Returns:
            List[Dict]: A list of dictionaries containing the search results, filtered based on the specified criteria.
        """
        raise NotImplementedError("Keyword search is not implemented for this vector store.")