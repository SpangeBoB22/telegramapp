import asyncio
import json
import logging
import os
import pathlib
import threading
import tracemalloc
from datetime import datetime, timezone, timedelta
from typing import Dict, List

from telethon import TelegramClient, events

from .. import conf
from .async_logger_wrap import AsyncLogger
from .crypto_util import JsonEncryptor
from ..core.es_connector import AsyncElasticsearchConnector
from app.core.models import API_data, ChannelMessage

tracemalloc.start()

logger = AsyncLogger()


class TelegramWorker:
    """

    A class designed to interact with Telegram using the Telethon library, performing
    various tasks such as fetching lists of channels, reading messages from specified channels,
    and indexing those messages into Elasticsearch for further processing or analysis.

    The class encapsulates the logic for initializing a Telegram client session, retrieving
    channel information, and processing incoming messages in real-time or from a specified
    time frame. It relies on external configuration and encryption utilities to securely
    handle API credentials and channel data.

    Attributes:
        _api_id (int): The Telegram API ID used for authentication.
        _api_hash (str): The Telegram API hash used for authentication.
        _session_name (str): The name of the Telegram session, used for persisting session data.
        _session_path (pathlib.Path): The file path to the session data.
        _es (AsyncElasticsearchConnector): An instance of AsyncElasticsearchConnector for indexing messages.

    Methods:
        __init__: Initializes a new TelegramWorker instance with session details and API credentials.
        get_channels_list: Static method that retrieves a list of Telegram channels.
        start_reading_channels: Static method that begins reading messages from configured channels.
        _list_channels: Asynchronously lists channels available to the session.
        _get_api_data: Static method that retrieves API data for a specified worker.
        _get_channels_list: Static method that fetches a list of channel IDs for a worker.
        read_the_channels: Asynchronously reads messages from specified channels.
        get_messages_from_today: Fetches messages from a channel for the current day.

    """

    def __init__(self, session_name: str, worker_id: int, api_id: int, api_hash: str) -> None:
        """

        Initializes a TelegramWorker instance with the necessary session details and API credentials
        for interacting with the Telegram API. It sets up the session path based on the provided session
        name and initializes the Elasticsearch connector for message indexing.

        Args:
            session_name (str): The name of the session for the Telegram client, used for session persistence.
            worker_id (int): The number of worker. Added for better logging
            api_id (int): The Telegram API ID, obtained from Telegram's Developer Portal.
            api_hash (str): The Telegram API hash, obtained alongside the API ID.

        Raises:
            ValueError: If any of the required initialization parameters are missing or invalid.

        """
        if not all([session_name, api_id, api_hash]):
            raise ValueError('Illegal init data for TelegramWorker')

        self._api_id = api_id
        self._api_hash = api_hash
        self._session_name = session_name
        self._session_path = conf.sessions_root / self._session_name
        self._es = AsyncElasticsearchConnector.get_aes_instance()
        self._queue = asyncio.Queue()
        self._worker_id = worker_id

    @property
    def worker_id(self):
        """

        The property which returns worker ID

        """
        return self._worker_id

    @staticmethod
    def get_channels_list(worker_id: int, password: str, file_to_save: pathlib.Path | None = None) -> None:
        """
        Retrieves a list of Telegram channels available to the worker's session and optionally saves
        the list to a specified file. If no file is specified, the list is printed to stdout.

        This method decrypts the worker's API data using the provided password to authenticate with
        Telegram and fetch the channel list.

        Args:
            worker_id (int): The identifier of the worker for whom to fetch the channel list.
            password (str): The password used to decrypt the worker's API data.
            file_to_save (pathlib.Path | None, optional): The file path where the channels list should be saved.
                If None, the list is printed to stdout.
        """
        api_data = TelegramWorker._get_api_data(worker_id, password)
        worker = TelegramWorker(api_data.session_name,
                                worker_id,
                                api_data.api_id,
                                api_data.api_hash)

        logger.log(logging.INFO, f"Retrieving channels list for worker {worker_id}")
        channels_dict = asyncio.run(worker._list_channels())
        lines = [f'ID: {k}, name: {v}' for k, v in channels_dict.items()]

        if file_to_save is not None:
            with open(file_to_save, 'w') as f:
                f.writelines(lines)
        else:
            for line in lines:
                print(line)

    @staticmethod
    def start_reading_all_channels(password: str) -> None:
        """

        Initiates the reading of all channels listed in the configuration file using multiple threads.
        For each channel, a separate thread is created where messages are read using the static method
        `start_reading_channels` of the `TelegramWorker` class. This parallel execution allows for efficient
        processing of multiple channels simultaneously.

        Each thread is responsible for connecting to a specific Telegram channel and retrieving messages
        based on the provided password for authentication. The method ensures that all channels are processed
        concurrently, improving the overall speed and efficiency of the operation.

        Args:
            password (str): The password used for authentication to access each channel's data.

        Raises:
            logging.ERROR: Logs an error message if an exception occurs during the start-up process
            of reading channels in the thread pool.

        Notes:
            This method leverages threading to create a non-blocking, concurrent execution environment for
            reading multiple Telegram channels. It waits for all threads to complete their execution before
            returning, ensuring that all channels are fully processed. Any errors encountered during the
            execution of threads are logged for debugging purposes.

        """
        with open(conf.channels_list_file) as f:
            data = json.load(f)
        workers = list(data.keys())

        try:
            threads = []
            for worker in workers:
                thread = threading.Thread(target=TelegramWorker.start_reading_channels, args=(worker, password, ))
                threads.append(thread)
                thread.start()

            [thread.join() for thread in threads]

        except Exception as ex:
            logger.log(logging.ERROR, message='Error while starting reading channels in threads', ex=ex)

    @staticmethod
    def start_reading_channels(worker_id: int, password: str):
        """
        Starts the process of reading messages from configured Telegram channels for the current day,
        using the specified worker's credentials. Messages are indexed into Elasticsearch for further analysis.

        This method initializes a Telegram client session using the decrypted API data and begins listening
        for new messages in real-time, as well as fetching past messages from the start of the current day.

        Args:
            worker_id (int): The identifier of the worker for whom to start reading messages.
            password (str): The password used to decrypt the worker's API data.
        """
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        api_data = TelegramWorker._get_api_data(worker_id, password)
        worker = TelegramWorker(api_data.session_name,
                                worker_id,
                                api_data.api_id,
                                api_data.api_hash)
        try:
            logger.log(level=logging.INFO,
                       message=f'Starting retrieving messages with {worker.worker_id}')

            channels = worker._get_channels_list(worker_id)

            loop.create_task(worker._run_queue_consumer())
            loop.run_until_complete(worker._read_the_channels(list_of_channels=channels))
        except Exception as ex:
            print(ex)
            logger.log(logging.ERROR,
                       message=f'Error while starting readings channels in worker ID {worker._worker_id}',
                       ex=ex)
        finally:
            loop.close()

    async def _run_queue_consumer(self):
        """

        Asynchronously consumes messages from a queue and indexes them into Elasticsearch.

        This internal method continuously consumes messages from an asyncio queue. Depending on the
        configuration, it either indexes each message individually or accumulates them into batches
        for bulk indexing. For batch processing, it creates an asynchronous task for each batch or
        individual message to be indexed in Elasticsearch. This method supports dynamic switching
        between batch and individual indexing modes based on configuration.

        The method leverages the asynchronous nature of the I/O operations involved in sending data
        to Elasticsearch, allowing for efficient processing of incoming messages without blocking the
        main event loop.

        Note:
            This method is designed to run indefinitely as a background task in an asyncio event loop.
            It automatically handles the creation of new tasks for the Elasticsearch indexing operations,
            ensuring that the queue is continuously consumed. The caller should ensure proper exception
            handling is in place to manage potential issues during the indexing process.

        """
        batch = [] if conf.es_use_batch else None
        loop = asyncio.get_event_loop()
        while True:
            message: ChannelMessage = await self._queue.get()
            self._queue.task_done()
            if conf.es_use_batch:
                batch.append(message)
                if len(batch) == conf.es_batch_size:
                    loop.create_task(self._es.write_batch(index=conf.es_index_name,
                                                          batch=batch))
                    batch = []
            else:
                loop.create_task(self._es.write_document(index=conf.es_index_name,
                                                         document=message))

    @staticmethod
    def _get_api_data(worker_id: int, password: str) -> API_data:
        """

         Retrieves the API data for a specific worker by decrypting the stored API data file with the given password.
         The method returns an API_data object containing the session name, API ID, and API hash for the worker.

         Args:
             worker_id (int): The identifier of the worker for whom to fetch API data.
             password (str): The password used to decrypt the stored API data.

         Returns:
             API_data: An object containing the decrypted API credentials for the specified worker.

         Raises:
             FileNotFoundError: If the API data file does not exist or cannot be found.
             ValueError: If the API data for the specified worker is not present in the file.

         """
        if not conf.api_data_file.exists():
            raise FileNotFoundError('API data file does not exist')

        data = JsonEncryptor().decrypt_json(conf.api_data_file, password)
        if str(worker_id) not in data:
            raise ValueError(f'Worker {worker_id} data not found')

        return API_data(session_name=data[str(worker_id)].get('SESSION_NAME'),
                        api_id=data[str(worker_id)].get('API_ID'),
                        api_hash=data[str(worker_id)].get('API_HASH'))

    @staticmethod
    def _get_channels_list(worker_id: int) -> List[int]:
        """
        Fetches a list of Telegram channel IDs configured for the specified worker from a stored JSON file.

        Args:
            worker_id (int): The identifier of the worker for whom to fetch the list of channel IDs.

        Returns:
            List[int]: A list of Telegram channel IDs associated with the specified worker.
        """
        with open(conf.channels_list_file) as f:
            data = json.load(f)
        return [int(value) for value in data[str(worker_id)]]

    async def _get_messages_from_today(self, client, channel) -> None:
        """
        Fetches messages from the specified Telegram channel that were posted since the start of the current day.
        This method asynchronously iterates over messages, creating tasks for each message to be indexed in
        Elasticsearch without waiting for each indexing operation to complete.

        Args:
            client (TelegramClient): The authenticated Telegram client session.
            channel ([str, int]): The identifier of the Telegram channel from which to fetch messages.
        """
        today = datetime.now(timezone.utc)
        loop = asyncio.get_event_loop()
        begin_of_the_day = (today.replace(hour=0, minute=0, second=0, microsecond=0)
                            - timedelta(days=conf.days_depth))

        loop.create_task(logger.alog(level=logging.INFO,
                                     message=f'Dumping today\'s messages from {channel}, worker ID {self.worker_id}'))
        try:
            async for message in client.iter_messages(channel, offset_date=today):

                if message.date < begin_of_the_day:
                    break
                if (message.text != "") | (message.message != ""):
                    self._queue.put_nowait(ChannelMessage(message_id=message.id,
                                                          publish_date=message.date,
                                                          source=message.chat.title,
                                                          message_text=message.text
                                                          if message.text != "" else message.message)
                                           )
        except Exception as ex:
            print(ex)
            loop.create_task(logger.alog(level=logging.CRITICAL,
                                         message=f'Exception while retrieving messages from {channel}'
                                                 f'in worker ID {self.worker_id}',
                                         ex=ex))
        finally:
            loop.create_task(logger.alog(level=logging.INFO,
                                         message=f'Dumping today\'s messages from {channel} is finished'
                                                 f' for worker ID {self.worker_id}'))

    async def _list_channels(self) -> Dict[int, str]:
        """
        Asynchronously retrieves a list of Telegram channels available to the authenticated session,
        returning a dictionary mapping channel IDs to their names.

        Returns:
            Dict[int, str]: A dictionary where keys are channel IDs and values are channel names.
        """
        channels_dict = {}
        async with TelegramClient(str(self._session_path), self._api_id, self._api_hash) as client:
            async for dialog in client.iter_dialogs():
                if dialog.is_channel:
                    channels_dict.update({dialog.id: dialog.name})
        return channels_dict

    async def _read_the_channels(self, list_of_channels: List[str | int] | None = None) -> None:
        """
        Asynchronously starts reading and processing messages from the specified list of Telegram channels.
        If no list is provided, it defaults to a pre-configured test channel. This method sets up event
        handlers for incoming messages and fetches messages from the start of the current day.

        Args:
            list_of_channels (List[str | int] | None, optional): A list of channel identifiers (names or IDs).
                Defaults to None, in which case a pre-configured test channel ID is used.
        """
        channel_ids = [int(os.environ.get("TEST_CHANNEL_ID"))] if list_of_channels is None else list_of_channels

        loop = asyncio.get_event_loop()
        loop.create_task(logger.alog(level=logging.INFO,
                                     message=f'Channels list worker ID {self.worker_id}: {channel_ids}'))

        async with TelegramClient(str(self._session_path), self._api_id, self._api_hash) as client:
            tasks = []
            for channel in channel_ids:
                task = asyncio.create_task(self._get_messages_from_today(client, channel))
                tasks.append(task)

            await asyncio.gather(*tasks)

            loop.create_task(logger.alog(level=logging.INFO,
                                         message=f'Start monitoring channels in worker ID {self.worker_id}'))

            @client.on(events.NewMessage(chats=channel_ids))
            async def handler(event):
                current_loop = asyncio.get_event_loop()
                try:
                    message = event.message
                    current_loop.create_task(logger.alog(level=logging.DEBUG,
                                                         message=f'New message received in {message.chat.title},'
                                                         f' worker ID {self.worker_id}'))

                    if (message.text != "") | (message.message != ""):
                        self._queue.put_nowait(ChannelMessage(message_id=message.id,
                                                              publish_date=message.date,
                                                              source=message.chat.title,
                                                              message_text=message.text
                                                              if message.text != "" else message.message)
                                               )
                except Exception as ex:
                    print(ex)
                    current_loop.create_task(logger.alog(level=logging.CRITICAL,
                                                         message=f'Error while processing message in {message.chat.title}'
                                                         f'Message ID: {message.id}, worker ID: {self.worker_id}',
                                                         ex=ex))

            await client.run_until_disconnected()
