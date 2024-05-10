import asyncio
from unittest import TestCase

from app.core.es_connector import AsyncElasticsearchConnector


class TestAsyncElasticsearchConnector(TestCase):

    def test_search_by_source(self):
        response = asyncio.run(AsyncElasticsearchConnector().get_documents_by_source('Forbes Ukraine'))
        pass

    def test_unique_sources(self):
        response = asyncio.run(AsyncElasticsearchConnector().get_unique_source_values())
        pass