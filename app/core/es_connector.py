import asyncio
import atexit
import json
import logging
import os
import pathlib
import ssl
from asyncio import Semaphore
from datetime import date
from typing import List, Dict, Tuple, Any

import aiofiles
from elasticsearch import AsyncElasticsearch
from elasticsearch.helpers import async_bulk, async_scan
from tqdm.asyncio import tqdm

from app import conf
from app.core import Singleton
from app.core.helpers import FileGenerator
from app.core.helpers import sanitize_string
from app.core.models import ChannelMessage
from app.core.models import ProcessedMessage
from app.tg_spider.async_logger_wrap import AsyncLogger

logger = AsyncLogger()

INDEX_MAPPING = {
    "settings": {
        "number_of_shards": 3,
        "number_of_replicas": 0
    },
    "mappings": {
        "properties": {
            "publish_date": {"type": "date"},
            "source": {"type": "keyword"},
            "initial_text": {"type": "text"},
            "cleared_text": {"type": "text"},
            "named_entities": {"type": "keyword"},
            "summary": {"type": "text"},
            "tonality": {"type": "keyword"},
            "category": {"type": "keyword"},
            "embedding_sts": {
                "type": "dense_vector",
                "dims": 768
            },
            "embedding_clf": {
                "type": "dense_vector",
                "dims": 768
            }
        }
    }
}


class AsyncElasticsearchConnector(metaclass=Singleton):
    """

    An asynchronous connector to Elasticsearch.
    Allows to write docs to the defined index asynchronously

    """

    def __init__(self):
        """
        Initializes an asynchronous connector to Elasticsearch.

        This connector is designed to facilitate communication with an Elasticsearch cluster
        using asynchronous operations. It supports both secure (https) and insecure (http) connections.
        The choice between http and https is determined by the `scheme` argument. For secure connections,
        it uses a default SSL context which ignores hostname and certificate verification to simplify
        development and testing scenarios. It's highly recommended to adjust SSL settings for production
        environments to ensure secure communication.

        Raises:
            ValueError: If no correct API key for Elasticsearch connection is provided.

        Note:
            The actual connection to Elasticsearch is not established in the constructor but upon
            performing an operation such as indexing a document or closing the connector.
        """
        self._ssl_context = ssl.create_default_context()
        self._ssl_context.check_hostname = False
        self._ssl_context.verify_mode = ssl.CERT_NONE

        self.__id, self.__key = conf.es_api_key

        self.es = AsyncElasticsearch(
            api_key=(self.__id, self.__key),
            hosts=[f'{conf.es_scheme}://{conf.es_host}:{conf.es_port}'],
            ssl_context=self._ssl_context,
            timeout=60
        )
        self._semaphore = Semaphore(conf.es_n_connections)
        atexit.register(self.close_connection)

    @staticmethod
    def get_aes_instance() -> "AsyncElasticsearchConnector":
        aes = AsyncElasticsearchConnector()
        loop = asyncio.get_event_loop()
        loop.run_until_complete(aes._ensure_index_with_mapping(aes.es, conf.es_index_name))
        return aes

    def close_connection(self) -> None:
        """
        Closes the connection on app's exit (atexit is used)
        """
        asyncio.run(self.es.close())

    async def write_document(self, index: str, document: ChannelMessage) -> Dict | None:
        """
        Asynchronously indexes a document to the specified Elasticsearch index.

        This method sends a document to the specified index in the Elasticsearch cluster.
        The operation is performed asynchronously, and the method returns a response object
        from Elasticsearch indicating the result of the index operation.

        Args:
            index (str): The name of the index where the document will be stored.
            document (ChannelMessage): The document to be indexed, represented as a dataclass' instance.

        Returns:
            dict: The response from Elasticsearch after indexing the document.

        Note:
            The method constructs an Elasticsearch 'index' operation and awaits its completion.
            The caller should handle any exceptions that may arise from this operation.
        """

        async with self._semaphore:
            try:
                body = document.get_body()
                response = await self.es.update(index=index,
                                                id=document.id_,
                                                doc=body,
                                                doc_as_upsert=True)
                return response
            except Exception as ex:
                loop = asyncio.get_event_loop()
                loop.create_task(logger.alog(logging.CRITICAL,
                                             message=f'Error while writing document to ES. Document ID: {document.id_}',
                                             ex=ex))
                return None

    async def write_batch(self, index: str, batch: List[ChannelMessage] | List[ProcessedMessage]) -> Tuple[int, int |
                                                                                                                List[
                                                                                                                    Any]] | None:
        """
        Asynchronously indexes a batch of documents to the specified Elasticsearch index with upsert logic.

        This method sends a batch of documents to the specified index in the Elasticsearch cluster.
        The operation is performed asynchronously, leveraging the bulk API for efficiency. Each document
        in the batch is indexed using the upsert logic, where documents are updated if they exist or inserted
        if they do not.

        Args:
            index (str): The name of the Elasticsearch index where the documents will be stored.
            batch (List[ChannelMessage]): A list of documents to be indexed, each represented as an instance of a dataclass.

        Returns:
            dict: The response from Elasticsearch after performing the bulk index operation.

        Note:
            The method constructs an Elasticsearch 'bulk' operation with upsert logic for each document
            in the batch and awaits its completion. The caller should handle any exceptions that may arise
            from this operation.
        """
        async with self._semaphore:
            try:
                bulk_body = []
                for document in batch:
                    document_body = {"_index": index,
                                     "_id": document.id_,
                                     "_source": document.get_body(),
                                     "doc_as_upsert": True}
                    bulk_body.append(document_body)
                loop = asyncio.get_event_loop()
                loop.create_task(logger.alog(logging.INFO,
                                             message=f'Documents to be indexed: {len(bulk_body)}'))

                response = await async_bulk(self.es, bulk_body, chunk_size=100)
                return response
            except Exception as ex:
                loop = asyncio.get_event_loop()
                loop.create_task(logger.alog(logging.CRITICAL,
                                             message=f'Error while writing batch into ES',
                                             ex=ex))
                return None

    @staticmethod
    async def _ensure_index_with_mapping(es, index_name: str):
        """
        Checks for the existence of an index in Elasticsearch and creates it with a predefined mapping
        if it does not exist. This function ensures that your application can work with a well-defined
        index structure, which is crucial for consistent data handling and querying.

        Args:
            es (AsyncElasticsearch): An instance of the Elasticsearch async client.
            index_name (str): The name of the index to check or create.

        This function first checks if the specified index exists using the `exists` method of the
        Elasticsearch indices API. If the index does not exist, it is created with the structure
        defined in the `INDEX_MAPPING` dictionary using the `create` method. This ensures that your
        index is ready to use with the correct settings and mappings for your data.

        Note:
            - The `INDEX_MAPPING` dictionary needs to be predefined and should match the structure
              of the data you plan to store in the index.
            - This function is asynchronous and should be called within an async context.
        """
        try:
            index_exists = await es.indices.exists(index=index_name)
        except Exception as ex:
            loop = asyncio.get_event_loop()
            loop.create_task(logger.alog(logging.CRITICAL,
                                         message=f'Error while checking index {index_name}',
                                         ex=ex))
            return None

        if not index_exists:
            loop = asyncio.get_event_loop()
            await es.indices.create(index=index_name, body=INDEX_MAPPING)
            loop.create_task(logger.alog(level=logging.INFO,
                                         message=f"Index {index_name} has been created."))

    def save_es_index_to_files(self, save_path: pathlib.Path, source: str | None = None):
        """
        Save Elasticsearch index documents to individual JSONL files in the specified directory.

        This method saves documents from the Elasticsearch index to individual JSONL files in the specified
        directory. If the directory exists, its contents are cleared before saving the documents. If the
        directory does not exist, it is created.

        Args:
            save_path (pathlib.Path): The path to the directory where the JSONL files will be saved.
            source (str, optional): The name of the source to filter documents. If None, all documents
                                    will be saved. Defaults to None.

        Returns:
            None
        """
        if save_path.exists() and save_path.is_dir():
            for filename in os.listdir(save_path):
                file_path = save_path / filename
                if file_path.is_file():
                    os.unlink(file_path)
        else:
            os.makedirs(save_path)

        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        loop.run_until_complete(self.async_save_es_index_to_files(save_path, source))

    async def async_save_es_index_to_files(self, save_path: pathlib.Path, source: str | None):
        """
        Asynchronously save Elasticsearch index documents to individual JSONL files.

        This method asynchronously saves documents from the Elasticsearch index to individual JSONL files
        in the specified directory. If the source parameter is provided, only documents from the specified
        source will be saved.

        Args:
            save_path (pathlib.Path): The path to the directory where the JSONL files will be saved.
            source (str, optional): The name of the source to filter documents. If None, all documents
                                    will be saved. Defaults to None.

        Returns:
            None
        """
        if source is None:
            query = {"query": {"match_all": {}}}
        else:
            query = {"query": {"match": {"source": source}}}

        async for hit in async_scan(client=self.es,
                                    index=conf.es_index_name,
                                    query=query):
            source_value = hit.get('_source', {}).get('source', 'unknown')
            sanitized_source_value = sanitize_string(source_value)
            file_path = save_path / f"{sanitized_source_value}.jsonl"

            async with aiofiles.open(file_path, 'a', encoding='utf') as file:
                await file.write(json.dumps(hit['_source'], ensure_ascii=False) + '\n')

    def load_from_files(self, file_path: pathlib.Path):
        """
        Load documents from JSONL files into the Elasticsearch index.

        This method loads documents from JSONL files into the Elasticsearch index. It reads documents
        from the specified directory and indexes them into Elasticsearch.

        Args:
            file_path (pathlib.Path): The path to the directory containing the JSONL files.

        Returns:
            None
        """
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        loop.run_until_complete(self.async_load_files_to_es_index(file_path))

    async def async_load_files_to_es_index(self, file_path: pathlib.Path):
        """
        Asynchronously load documents from JSONL files into the Elasticsearch index.

        This method asynchronously loads documents from JSONL files into the Elasticsearch index.
        It reads documents from the specified directory and indexes them into Elasticsearch.

        Args:
            file_path (pathlib.Path): The path to the directory containing the JSONL files.

        Returns:
            None
        """
        tasks = []
        async for file_path in FileGenerator(file_path):
            tasks.append(self._async_index_file(self.es, file_path))
        await tqdm.gather(*tasks, total=len(tasks), desc='Indexing files')

    @staticmethod
    async def _async_index_file(session: AsyncElasticsearch,
                                file_path: pathlib.Path):
        """
        Asynchronously index documents from a JSONL file into the Elasticsearch index.

        This method asynchronously indexes documents from a JSONL file into the Elasticsearch index.
        It reads documents from the specified file and bulk indexes them into Elasticsearch.

        Args:
            session (AsyncElasticsearch): An instance of the Elasticsearch async client.
            file_path (pathlib.Path): The path to the JSONL file containing the documents.

        Returns:
            None
        """
        async with aiofiles.open(file_path, 'r', encoding='utf') as file:
            bulk_body = []
            async for line in file:
                doc = ProcessedMessage.from_jsonl_dump(line)
                if doc:
                    document_body = {"_index": conf.es_index_name,
                                     "_id": doc.message_id,
                                     "_source": doc.get_body(),
                                     "doc_as_upsert": True}
                    bulk_body.append(document_body)

        await async_bulk(session, bulk_body, chunk_size=100)

    async def get_documents_by_source(self, source_value: str) -> List[ProcessedMessage]:
        """
        Retrieves all documents from the index with the specified source field value.

        Args:
            source_value: The value of the source field to filter documents by.

        Returns:
            A list of documents satisfying the specified condition.
        """
        query = {
            "query": {
                "match": {
                    "source": source_value
                }
            }
        }
        result = await self.es.search(index=conf.es_index_name, body=query, size=10000)
        documents = result['hits']['hits']
        response = [ProcessedMessage.from_es_source(doc['_id'], doc['_source']) for doc in documents]
        return response

    async def get_all_documents(self) -> List[ProcessedMessage]:
        """
        Retrieve all documents from the Elasticsearch index.

        This method retrieves all documents from the Elasticsearch index.

        Returns:
            List[ProcessedMessage]: A list of ProcessedMessage objects representing the retrieved documents.
        """
        raise NotImplementedError

    async def get_unprocessed_chunk(self) -> List[ProcessedMessage]:
        """
        Retrieve an unprocessed chunk of documents from the Elasticsearch index.

        This method retrieves an unprocessed chunk of documents from the Elasticsearch index. It queries
        documents where the 'cleared_text' field is empty.

        Returns:
            List[ProcessedMessage]: A list of ProcessedMessage objects representing the unprocessed documents.
        """
        query = {
            "size": conf.es_process_chunk,
            "query": {
                "bool": {
                    "must": {
                        "match_all": {}
                    },
                    "must_not": {
                        "exists": {
                            "field": "cleared_text"
                        }
                    }
                }
            }
        }

        result = await self.es.search(index=conf.es_index_name, body=query)
        documents = result['hits']['hits']
        response = [ProcessedMessage.from_es_source(doc['_id'], doc['_source']) for doc in documents]
        return response

    async def get_documents_by_date_range(self, start_date: date, end_date: date) -> List[ProcessedMessage]:
        """
        Retrieves all documents from the index within the specified date range.

        Args:
            start_date: The start date of the date range (Python date object).
            end_date: The end date of the date range (Python date object).

        Returns:
            A list of documents falling within the specified date range.
        """
        async with self.es as es:
            query = {
                "query": {
                    "range": {
                        "timestamp": {
                            "gte": start_date.isoformat(),
                            "lte": end_date.isoformat()
                        }
                    }
                }
            }
            result = await es.search(index=conf.es_index_name, body=query, size=10000)
            documents = result['hits']['hits']
        response = [ProcessedMessage.from_es_source(doc['_id'], doc['_source']) for doc in documents]
        return response

    async def get_unique_source_values(self) -> list:
        """
        Retrieves all unique values of the 'source' field from the index.

        Returns:
            A list of all unique values of the 'source' field.
        """
        async with self.es as es:
            aggregation_query = {
                "size": 0,
                "aggs": {
                    "unique_sources": {
                        "terms": {
                            "field": "source.keyword",
                            "size": 10000
                        }
                    }
                }
            }
            result = await es.search(index=conf.es_index_name, body=aggregation_query)
        unique_values = [bucket['key'] for bucket in result['aggregations']['unique_sources']['buckets']]
        return unique_values
