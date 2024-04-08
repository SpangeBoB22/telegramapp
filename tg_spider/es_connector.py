import asyncio
import json
import logging
import os
import pathlib
import re
import ssl
from asyncio import Semaphore
from typing import List, Dict, Tuple, Any

import aiofiles
from elasticsearch import AsyncElasticsearch
from elasticsearch.helpers import async_bulk, async_scan

from tg_spider import conf
from tg_spider.async_logger_wrap import AsyncLogger
from tg_spider.core import Singleton
from tg_spider.models import ChannelMessage

logger = AsyncLogger()

INDEX_MAPPING = {
        "settings": {
            "number_of_shards": 2,
            "number_of_replicas": 0
        },
        "mappings": {
            "properties": {
                "publish_date": {"type": "date"},
                "source": {"type": "keyword"},
                "text": {"type": "text"}
            }
        }
    }


LOG_INDEX_MAPPING = {
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0
    },
    "mappings": {
        "properties": {
            "timestamp": {"type": "date"},
            "level": {"type": "keyword"},
            "message": {"type": "text"},
            "application": {"type": "keyword"},
            "extra_data": {"type": "object", "enabled": False}
        }
    }
}


class AsyncElasticsearchConnector(metaclass=Singleton):

    """

    An asynchronous connector to Elasticsearch.
    Allows to write docs to the defined index asynchronously
    Sample of use:

    >>> import asyncio
    >>> from tg_spider.es_connector import AsyncElasticsearchConnector
    >>> from datetime import datetime, timezone

    >>>es = AsyncElasticsearchConnector()
    >>>cluster_health = asyncio.run(es.es.cluster.health())
    >>>print("Cluster Health:", cluster_health)

    >>>now_utc = datetime.now(timezone.utc)
    >>>iso_format_date = now_utc.isoformat()

    >>>asyncio.run(es.write_document(conf.es_index_name, document={'date': iso_format_date,
                                                                   'channel': 'Sample channel name',
                                                                   'text': 'Sample text'}))
    >>>asyncio.run(es.close())

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
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE

        id_, key = conf.es_api_key

        if id_ is not None:
            self.es = AsyncElasticsearch(
                api_key=(id_, key),
                hosts=[f'{conf.es_scheme}://{conf.es_host}:{conf.es_port}'],
                ssl_context=ssl_context,
                timeout=60
            )
        else:
            raise ValueError('No correct key for Elasticsearch connection provided.')

        loop = asyncio.get_event_loop()
        loop.run_until_complete(self._ensure_index_with_mapping(self.es, conf.es_index_name))

        self._semaphore = Semaphore(conf.es_n_connections)

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

    async def write_batch(self, index: str, batch: List[ChannelMessage]) -> Tuple[int, int | List[Any]] | None:
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

    async def close(self):
        """
        Closes the asynchronous connection to Elasticsearch.

        This method properly closes the connection to the Elasticsearch cluster by
        shutting down the underlying asynchronous HTTP session. It's important to call
        this method before exiting the application to ensure all resources are released.

        Note:
            Always await this method to ensure the connection is closed properly.
        """
        await self.es.close()

    @staticmethod
    async def _ensure_index_with_mapping(es: AsyncElasticsearch, index_name: str):
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

    def save_es_index_to_files(cls, save_path: pathlib.Path):
        # Перевірка та очищення директорії зберігання
        if save_path.exists() and save_path.is_dir():
            for filename in os.listdir(save_path):
                file_path = save_path / filename
                if file_path.is_file():
                    os.unlink(file_path)
        else:
            # Створення директорії, якщо вона не існує
            os.makedirs(save_path)

        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        loop.run_until_complete(cls.async_save_es_index_to_files(save_path))

    async def async_save_es_index_to_files(self, save_path: pathlib.Path):

        # Виконання асинхронного пошуку для отримання усіх записів з індексу
        async for hit in async_scan(client=self.es,
                                    index=conf.es_index_name,
                                    query={"query": {"match_all": {}}}):
            source_value = hit.get('_source', {}).get('source', 'unknown')
            sanitized_source_value = self._sanitize_filename(source_value)
            file_path = save_path / f"{sanitized_source_value}.jsonl"

            # Асинхронне додавання даних до файлу
            async with aiofiles.open(file_path, 'a', encoding='utf') as file:
                await file.write(json.dumps(hit['_source'], ensure_ascii=False) + '\n')

        await self.es.close()

    @staticmethod
    def _sanitize_filename(filename):
        """
        Видаляє недопустимі символи з імен файлів.
        """
        # Замінює будь-який недопустимий символ на підкреслення
        return re.sub(r'[\/\\:*?"<>|]', '_', filename)



