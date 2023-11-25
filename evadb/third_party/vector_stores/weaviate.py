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
import os
from typing import List, Dict

from evadb.third_party.vector_stores.types import (
    FeaturePayload,
    VectorIndexQuery,
    VectorIndexQueryResult,
    VectorStore,
    ObjectUpdatePayload,
    ObjectReplacePayload,
    ObjectPropertyDeletionPayload,
    SearchQueryPayload,
    GenerativeSearchPayload,
    LabelBasedSearchPayload
)
from evadb.utils.generic_utils import try_to_import_weaviate_client

required_params = []
_weaviate_init_done = False


class WeaviateVectorStore(VectorStore):
    def __init__(self, collection_name: str, **kwargs) -> None:
        try_to_import_weaviate_client()
        global _weaviate_init_done

        self._collection_name = collection_name

        # Get the API key.
        self._api_key = kwargs.get("WEAVIATE_API_KEY")

        if not self._api_key:
            self._api_key = os.environ.get("WEAVIATE_API_KEY")

        assert (
            self._api_key
        ), "Please set your `WEAVIATE_API_KEY` using set command or environment variable (WEAVIATE_API_KEY). It can be found at the Details tab in WCS Dashboard."

        # Get the API Url.
        self._api_url = kwargs.get("WEAVIATE_API_URL")

        if not self._api_url:
            self._api_url = os.environ.get("WEAVIATE_API_URL")

        assert (
            self._api_url
        ), "Please set your `WEAVIATE_API_URL` using set command or environment variable (WEAVIATE_API_URL). It can be found at the Details tab in WCS Dashboard."

        if not _weaviate_init_done:
            # Initialize weaviate client
            import weaviate

            client = weaviate.Client(
                url=self._api_url,
                auth_client_secret=weaviate.AuthApiKey(api_key=self._api_key),
            )
            client.schema.get()

            _weaviate_init_done = True

        self._client = client

    def create(
        self,
        vectorizer: str = "text2vec-openai",
        properties: list = None,
        module_config: dict = None,
    ):
        properties = properties or []
        module_config = module_config or {}

        collection_obj = {
            "class": self._collection_name,
            "properties": properties,
            "vectorizer": vectorizer,
            "moduleConfig": module_config,
        }

        if self._client.schema.exists(self._collection_name):
            self._client.schema.delete_class(self._collection_name)

        self._client.schema.create_class(collection_obj)

    def add(self, payload: List[FeaturePayload]) -> None:
        with self._client.batch as batch:
            for item in payload:
                data_object = {"id": item.id, "vector": item.embedding}
                batch.add_data_object(data_object, self._collection_name)

    def delete(self) -> None:
        self._client.schema.delete_class(self._collection_name)

    def query(self, query: VectorIndexQuery) -> VectorIndexQueryResult:
        response = (
            self._client.query.get(self._collection_name, ["*"])
            .with_near_vector({"vector": query.embedding})
            .with_limit(query.top_k)
            .do()
        )

        data = response.get("data", {})
        results = data.get("Get", {}).get(self._collection_name, [])

        similarities = [item["_additional"]["distance"] for item in results]
        ids = [item["id"] for item in results]

        return VectorIndexQueryResult(similarities, ids)

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
        self._client.data_object.update(
            data_object=payload.properties,
            class_name=self._collection_name,
            uuid=payload.uuid
        )

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
        self._client.data_object.replace(
            data_object=payload.properties,
            class_name=self._collection_name,
            uuid=payload.uuid
        )

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

        # Fetch the current object data
        object_data = self._client.data_object.get_by_id(payload.uuid)['properties']

        # Remove the properties to delete
        updated_properties = {
            key: value for key, value in object_data.items() if key not in payload.properties_to_delete
        }

        # Replace the object with the updated properties
        self._client.data_object.replace(
            data_object=updated_properties,
            class_name=self._collection_name,
            uuid=payload.uuid
        )

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

        # Build the BM25 search query
        search_query = (
            self._client.query
            .get(self._collection_name, payload.properties)
            .with_bm25(query=payload.query, properties=payload.properties)
        )

        # Apply autocut if provided
        if payload.autocut is not None:
            search_query = search_query.with_autocut(payload.autocut)

        # Set the result limit and execute the query
        response = search_query.with_limit(payload.limit).do()

        data = response.get('data', {})

        # Extract the results
        results = data['Get'][self._collection_name]
        return VectorIndexQueryResult([], [], others=results)

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

        response = (
            self._client.query
            .get(self._collection_name, payload.properties)
            .with_hybrid(
                query=payload.query
            )
            .with_limit(payload.limit)
            .with_additional(["distance"])
            .do()
        )

        data = response.get('data', {})

        # Extract the results
        results = data['Get'][self._collection_name]
        return VectorIndexQueryResult([], [], others=results)

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
        # Build the generative search query
        search_query = (
            self._client.query
            .get(self._collection_name, payload.properties)
            .with_near_text({"concepts": [payload.query]})
        )

        # Apply generative search based on the specified prompt
        if payload.grouped_task:
            search_query = search_query.with_generate(
                grouped_task=payload.prompt,
                grouped_properties=payload.grouped_properties
            )
        else:
            search_query = search_query.with_generate(single_prompt=payload.prompt)

        # Set the result limit and execute the query
        response = search_query.with_limit(payload.limit).do()

        data = response.get('data', {})

        # Extract the results
        results = data['Get'][self._collection_name]
        return VectorIndexQueryResult([], [], others=results)

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
        response = (
            self._client.query
            .get(self._collection_name, payload.properties)
            .with_where(payload.filters)
            .with_limit(payload.limit)
            .do()
        )

        data = response.get('data', {})
        results = data.get('Get', {}).get(self._collection_name, [])
        return VectorIndexQueryResult([], [], others=results)
