import asyncio
from time import sleep, time
from typing import Literal, List

from app import conf
from app.core.es_connector import AsyncElasticsearchConnector
from app.core.models import ProcessedMessage
from app.news_explore.processor_llm import ProcessorLLM
from colorama import Fore, Style


class ProcessRunner:
    """A class to run various processing tasks on news channel documents."""

    @staticmethod
    def process_as_daemon():
        """
        Run the processing tasks continuously as a daemon.

        This method runs the processing tasks continuously in a loop as a daemon,
        fetching unprocessed documents in chunks from the Elasticsearch index,
        processing them to extract clear text, named entities, summaries,
        tonality, and topics, and then writing the processed documents back to the index.

        Returns:
            None
        """
        asyncio.run(ProcessRunner._process_as_daemon())

    @classmethod
    async def _process_as_daemon(cls):
        """
        Internal method to run processing tasks continuously as a daemon.

        This method fetches unprocessed documents in chunks from the Elasticsearch index,
        processes them to extract clear text, named entities, summaries, tonality, and topics,
        and then writes the processed documents back to the index.
        It runs in a loop indefinitely.

        Returns:
            None
        """
        while True:
            aes = AsyncElasticsearchConnector()
            llm = ProcessorLLM()

            docs = await aes.get_unprocessed_chunk()
            if len(docs) > 0:
                print(Fore.GREEN + f'{len(docs)} or more unprocessed documents were found.' + Style.RESET_ALL)

                docs = await llm.define_tonality(docs)
                docs = await llm.define_category(docs)
                docs = await llm.get_summary(docs)
                docs = await llm.get_clear_text_and_named_entities(docs)
                docs = await llm.get_embeddings(docs)

                await aes.write_batch(conf.es_index_name, docs)
            else:
                print(Fore.RED + 'No unprocessed documents were found.' + Style.RESET_ALL)
                sleep(3600)

    @staticmethod
    def process_clear_and_ner(channel_name: str | None):
        """
        Process documents from a single channel by clearing text and extracting named entities.

        Args:
            channel_name (str | None): The name of the news channel. If None, processes all channels.

        Returns:
            None
        """
        asyncio.run(ProcessRunner._process(channel_name, process_type='clear_and_ner'))

    @staticmethod
    def process_summary(channel_name: str | None = None):
        """
        Generate summaries for documents from a single channel.

        Args:
            channel_name (str | None): The name of the news channel. If None, processes all channels.

        Returns:
            None
        """
        asyncio.run(ProcessRunner._process(channel_name, process_type='summary'))

    @staticmethod
    def process_tonality(channel_name: str | None = None):
        """
        Define tonality for documents from a single channel.

        Args:
            channel_name (str | None): The name of the news channel. If None, processes all channels.

        Returns:
            None
        """
        asyncio.run(ProcessRunner._process(channel_name, process_type='tonality'))

    @staticmethod
    def generate_embeddings(channel_name: str | None = None):
        asyncio.run(ProcessRunner._process(channel_name, 'embeddings'))

    @staticmethod
    def process_all(channel_name: str | None = None):
        async def process_all(channel_name: str | None = None):
            start_time = time()
            await ProcessRunner._process(channel_name, process_type='clear_and_ner')
            await asyncio.sleep(61 - start_time)
            start_time = time()
            await ProcessRunner._process(channel_name, process_type='summary')
            await asyncio.sleep(61 - start_time)
            start_time = time()
            await ProcessRunner._process(channel_name, process_type='tonality')
            await asyncio.sleep(61 - start_time)
            start_time = time()
            await ProcessRunner._process(channel_name, process_type='category')
            await asyncio.sleep(61 - start_time)
            await ProcessRunner._process(channel_name, process_type='embeddings')

        asyncio.run(process_all(channel_name))

    @staticmethod
    def process_category(channel_name: str | None = None):
        """
        Extract topics from documents from a single channel.

        Args:
            channel_name (str | None): The name of the news channel. If None, processes all channels.

        Returns:
            None
        """
        asyncio.run(ProcessRunner._process(channel_name, process_type='category'))

    @classmethod
    async def _process(cls,
                       channel_name: str | None,
                       process_type: Literal['clear_and_ner', 'summary', 'tonality', 'category', 'embeddings'],
                       reprocess_docs: bool = True,
                       exception_on_emtpy: bool = False):
        """
        Internal method to process documents based on the specified type.

        Args:
            channel_name (str | None): The name of the news channel. If None, processes all channels.
            process_type (Literal['clear_and_ner', 'summary', 'tonality', 'topics']): The type of processing.
            reprocess_docs (bool): Whether to reprocess documents.
            exception_on_emtpy (bool): Whether to raise an exception when documents sequence is empty.

        Raises:
            ValueError: If an unknown process type is provided.

        Returns:
            None
        """

        aes = AsyncElasticsearchConnector()

        if channel_name:
            docs = await aes.get_documents_by_source(channel_name)
            if len(docs) == 0:
                if exception_on_emtpy:
                    raise RuntimeError(f'No documents found for channel {channel_name}')
            print(f'{len(docs)} documents found')
            await cls._process_docs_sequence(aes, docs, process_type, reprocess_docs)
        else:
            sources = await aes.get_unique_source_values()
            for source in sources:
                print(f'Processing all sources. Current source: {source}')
                docs = await aes.get_documents_by_source(source)
                if len(docs) == 0:
                    if exception_on_emtpy:
                        raise RuntimeError(f'No documents found for channel {channel_name}')
                await cls._process_docs_sequence(aes, docs, process_type, reprocess_docs)

    @staticmethod
    async def _process_docs_sequence(aes: AsyncElasticsearchConnector,
                                     docs: List[ProcessedMessage],
                                     process_type: Literal[
                                         'clear_and_ner', 'summary', 'tonality', 'category', 'embeddings'],
                                     reprocess_docs: bool = True):
        """
        Internal method to process documents based on the specified type.

        Args:
            docs: List(ProcessedMessage): A list of docs to process
            process_type (Literal['clear_and_ner', 'summary', 'tonality', 'topics']): The type of processing.
            reprocess_docs (bool): Whether to reprocess documents.

        Raises:
            ValueError: If an unknown process type is provided.

        Returns:
            None
        """
        llm = ProcessorLLM()

        if not reprocess_docs:
            docs = list(filter(lambda doc: doc.cleared_text == '', docs))
            print(f'{len(docs)} will be processed')

        if len(docs) > 0:
            match process_type:
                case 'clear_and_ner':
                    updated_docs = await llm.get_clear_text_and_named_entities(docs)
                case 'summary':
                    updated_docs = await llm.get_summary(docs)
                case 'tonality':
                    updated_docs = await llm.define_tonality(docs)
                case 'category':
                    updated_docs = await llm.define_category(docs)
                case 'embeddings':
                    updated_docs = await llm.get_embeddings(docs)
                case _:
                    raise ValueError(f'Unknown process type: {process_type}')
            await aes.write_batch(conf.es_index_name, updated_docs)


if __name__ == '__main__':
    CHANNEL = 'hromadske'
    ProcessRunner.generate_embeddings(CHANNEL)
