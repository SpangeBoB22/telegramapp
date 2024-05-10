import re
from asyncio import sleep
from time import time
from typing import List, Callable, Literal, Union

from tqdm.asyncio import tqdm

from .llms.llm_model import LLM_Model
from .llms.prompts import Prompt
from .settings import MESSAGES_COUNT_THRESHOLD, SETUPS, N_OF_ASYNC_REQUESTS, EMBEDDINGS_THRESHOLD
from ..constants import BACKGROUND
from ..core import Singleton
from ..core.helpers import chunker, seconds_to_hms
from ..core.models import ModelSetup, ProcessedMessage

ACTION_PATTERN = r'Дія [1,2]:'
TEXT_PATTERN = r'Текст #\d{1,2}\s{0,1}:'


class ProcessorLLM(metaclass=Singleton):
    """
    Class for processing messages using Language Models.
    """

    @classmethod
    async def get_summary(
            cls,
            messages: ProcessedMessage | List[ProcessedMessage]) -> ProcessedMessage | List[ProcessedMessage]:
        """
        Get summaries for the provided messages.

        Args:
            messages (ProcessedMessage | List[ProcessedMessage]): A single ProcessedMessage object or a list of
                                                                   ProcessedMessage objects.

        Returns:
            ProcessedMessage | List[ProcessedMessage]: A single ProcessedMessage object or a list of
                                                       ProcessedMessage objects with summaries.
        """
        kwargs = {'prompt_func': Prompt.summary_chunk_prompt,
                  'setup': SETUPS['high']}

        if isinstance(messages, ProcessedMessage):
            return await cls._process_single_message(messages,
                                                     **kwargs)

        elif isinstance(messages, list):
            return await cls._process_multi_message(messages,
                                                    **kwargs
                                                    )
        else:
            raise ValueError('Messages must be a list or a ProcessedMessage')

    @classmethod
    async def get_clear_text_and_named_entities(
            cls,
            some_messages: ProcessedMessage | List[ProcessedMessage]) -> ProcessedMessage | List[ProcessedMessage]:
        """
        Get clear text and named entities for the provided messages.

        Args:
            some_messages (ProcessedMessage | List[ProcessedMessage]): A single ProcessedMessage object or a list of
                                                                       ProcessedMessage objects.

        Returns:
            ProcessedMessage | List[ProcessedMessage]: A single ProcessedMessage object or a list of
                                                       ProcessedMessage objects with clear text and named entities.
        """
        kwargs = {'prompt_func': Prompt.low_temp_prompt,
                  'setup': SETUPS['low']}
        if isinstance(some_messages, ProcessedMessage):
            return await cls._process_single_message(some_messages,
                                                     **kwargs)

        elif isinstance(some_messages, list):
            return await cls._process_multi_message(some_messages,
                                                    **kwargs
                                                    )
        else:
            raise ValueError('Messages must be a list or a ProcessedMessage')

    @classmethod
    async def get_embeddings(cls,
                             messages: List[ProcessedMessage]) -> List[ProcessedMessage]:
        return await cls._process_multi_message(messages,
                                                None,
                                                setup=SETUPS['embeddings'],
                                                embeddings=True)

    @staticmethod
    async def _process_single_message(message: ProcessedMessage,
                                      prompt_func: Callable,
                                      setup: ModelSetup) -> ProcessedMessage:
        """
        Process a single message asynchronously.

        Args:
            message (ProcessedMessage): The ProcessedMessage object to process.
            prompt_func (Callable): The prompt function to use for processing.
            setup (ModelSetup): The model setup to use for processing.

        Returns:
            ProcessedMessage: The processed ProcessedMessage object.
        """
        raise NotImplementedError

    @classmethod
    async def _process_multi_message(cls,
                                     messages: List[ProcessedMessage],
                                     prompt_func: Union[Callable, None],
                                     setup: ModelSetup,
                                     embeddings: bool = False) -> List[ProcessedMessage]:
        """
        Process multiple messages asynchronously.

        Args:
            messages (List[ProcessedMessage]): The list of ProcessedMessage objects to process.
            prompt_func (Callable): The prompt function to use for processing.

        Returns:
            List[ProcessedMessage]: The list of processed ProcessedMessage objects.
        """
        count = len(messages)
        messages_threshold = MESSAGES_COUNT_THRESHOLD[SETUPS['low'].name] \
            if not embeddings else EMBEDDINGS_THRESHOLD[SETUPS['embeddings'].name]
        async_count = N_OF_ASYNC_REQUESTS[SETUPS['low'].name] \
            if not embeddings else N_OF_ASYNC_REQUESTS[SETUPS['embeddings'].name]

        if count <= messages_threshold * async_count:
            if not embeddings:
                response = await cls._retrieve_async(messages, prompt_func, setup)
            else:
                response = await cls._retrieve_async(messages, None, setup, embeddings=True)
            return response
        else:
            response, elapsed_time = [], 100
            n_of_chunks = (len(messages) // (messages_threshold * async_count) +
                           (1 if (len(messages) % (messages_threshold * async_count)) != 0 else 0))
            for n, chunk in enumerate(chunker(messages, size=messages_threshold * async_count)):
                if elapsed_time <= 60:
                    await sleep(60 - elapsed_time)

                start_time = time()
                print(f'Chunk # {n + 1}/{n_of_chunks}.'
                      f' Start time: {seconds_to_hms(start_time)}.'
                      f' Docs in chunk: {len(chunk)}.'
                      f' Prompt: {str(prompt_func.__name__) if prompt_func is not None else 'embeddings'}')

                if not embeddings:
                    partial_response = await cls._retrieve_async(chunk, prompt_func, setup=setup)
                else:
                    partial_response = await cls._retrieve_async(chunk, None, setup=setup,
                                                                 embeddings=True)
                response.append(partial_response)

                print(f'\nProcessing of chunk# {n + 1} is finished. time: {seconds_to_hms(time())}')

                elapsed_time = time() - start_time

            return sum(response, [])

    @classmethod
    async def _retrieve_async(cls,
                              messages: List[ProcessedMessage],
                              prompt_func: Union[Callable, None],
                              setup: ModelSetup,
                              embeddings: bool = False) -> List[ProcessedMessage]:
        """
        Retrieve responses asynchronously.

        Args:
            messages (List[ProcessedMessage]): The list of ProcessedMessage objects.
            prompt_func (Callable): The prompt function to use for retrieving responses.

        Returns:
            List[ProcessedMessage]: The list of ProcessedMessage objects with responses.
        """
        async with LLM_Model(setup_model=setup) as llm:
            if not embeddings:
                grouped_messages, count_threshold = [], MESSAGES_COUNT_THRESHOLD[SETUPS['low'].name]
                for chunk in chunker(messages, count_threshold):
                    grouped_messages.append([item.initial_text for item in chunk])

                questions = [prompt_func(messages_chunk) for messages_chunk in grouped_messages]
                tasks = [llm.answer_async(background=BACKGROUND,
                                          question=question) for question in questions]
                response = await tqdm.gather(*tasks, desc=prompt_func.__name__) if tasks else []
            else:
                grouped_messages, count_threshold = [], EMBEDDINGS_THRESHOLD[SETUPS['embeddings'].name]

                for chunk in chunker(messages, count_threshold):
                    grouped_messages.append([item.cleared_text for item in chunk])
                tasks_sts = [llm.get_embeddings(message_group, 'similarity') for message_group in grouped_messages]
                tasks_clf = [llm.get_embeddings(message_group, 'classification') for message_group in grouped_messages]
                response_sts = await tqdm.gather(*tasks_sts) if tasks_sts else []
                response_clf = await tqdm.gather(*tasks_clf) if tasks_clf else []

        if not embeddings:
            response = [item if 'Error: 400' not in item else [None] * count_threshold for item in response]
            match prompt_func.__name__:
                case 'low_temp_prompt':
                    messages = cls._post_process_messages_clean_and_ner(messages, response)
                case 'summary_chunk_prompt':
                    messages = cls._post_process_messages_single_answer(messages, response, 'summary')
                case 'tonality_prompt':
                    messages = cls._post_process_messages_single_answer(messages, response, 'tonality')
                case 'category_prompt':
                    messages = cls._post_process_messages_single_answer(messages, response, 'category')
                case _:
                    raise ValueError('Unknown prompt type')
        else:
            response_sts = sum(
                [item if not isinstance(item, str) else [None] * count_threshold for item in response_sts], [])
            response_clf = sum(
                [item if not isinstance(item, str) else [None] * count_threshold for item in response_clf], [])
            for message, embedding_sts, embedding_clf in zip(messages, response_sts, response_clf):
                message.embedding_sts, message.embedding_clf = embedding_sts, embedding_clf
        return messages

    @classmethod
    def _post_process_messages_clean_and_ner(cls,
                                             messages: List[ProcessedMessage],
                                             response: List[str]) -> List[ProcessedMessage]:
        """
        Post-process messages for cleaning and Named Entity Recognition.

        Args:
            messages (List[ProcessedMessage]): The list of ProcessedMessage objects.
            response (List[str]): The list of response strings.

        Returns:
            List[ProcessedMessage]: The list of processed ProcessedMessage objects.
        """
        splitted_responses = [item.split('|||') if isinstance(item, str) else item for item in response]
        splitted_responses = sum(splitted_responses, [])
        splitted_responses = list(filter(lambda x: True if x is None else (x != ''), splitted_responses))
        return list(map(lambda x: cls._process_single_response_clean_and_ner(x[0], x[1]) if x[1] is not None else x[0],
                        zip(messages, splitted_responses)))

    @classmethod
    def _post_process_messages_single_answer(cls,
                                             messages: List[ProcessedMessage],
                                             response: List[str],
                                             field_name: Literal['summary', 'tonality', 'category']) \
            -> List[ProcessedMessage]:
        """
        Post-process messages for summarization, tonality, category.

        Args:
            messages (List[ProcessedMessage]): The list of ProcessedMessage objects.
            response (List[str]): The list of response strings.

        Returns:
            List[ProcessedMessage]: The list of processed ProcessedMessage objects.
        """
        parts = sum([re.split(TEXT_PATTERN, single_response) for single_response in response], [])
        parts = list(map(lambda x: x.strip(), list(filter(None, parts))))
        for message, field_value in zip(messages, parts):
            setattr(message, field_name, field_value.lower())
        return messages

    @staticmethod
    def _process_single_response_clean_and_ner(message: ProcessedMessage, response: str) -> ProcessedMessage:
        """
        Process a single response for cleaning and Named Entity Recognition.

        Args:
            message (ProcessedMessage): The ProcessedMessage object.
            response (str): The response string.

        Returns:
            ProcessedMessage: The processed ProcessedMessage object.
        """
        try:
            parts = re.split(ACTION_PATTERN, response.strip())

            if len(parts) >= 2:
                message.cleared_text = parts[1].strip()
            else:
                print(parts)
            if len(parts) >= 3:
                message.named_entities = list(map(str.strip, parts[2].split(';')))
            else:
                print(parts)
        except Exception as e:
            print(message)
            print(e)
        finally:
            message.named_entities = list(filter(lambda x: len(x) > 2, message.named_entities))
            return message

    @classmethod
    async def define_tonality(
            cls,
            messages: ProcessedMessage | List[ProcessedMessage]) -> ProcessedMessage | List[ProcessedMessage]:
        """
        Define tonality for the provided messages.

        Args:
            messages (ProcessedMessage | List[ProcessedMessage]): A single ProcessedMessage object or a list of
                                                                   ProcessedMessage objects.

        Returns:
            ProcessedMessage | List[ProcessedMessage]: A single ProcessedMessage object or a list of
                                                       ProcessedMessage objects with tonality defined.
        """
        kwargs = {'prompt_func': Prompt.tonality_prompt,
                  'setup': SETUPS['high']}

        if isinstance(messages, ProcessedMessage):
            return await cls._process_single_message(messages,
                                                     **kwargs)

        elif isinstance(messages, list):
            return await cls._process_multi_message(messages,
                                                    **kwargs
                                                    )
        else:
            raise ValueError('Messages must be a list or a ProcessedMessage')

    @classmethod
    async def define_category(
            cls,
            messages: ProcessedMessage | List[ProcessedMessage]) -> ProcessedMessage | List[ProcessedMessage]:
        """
        Define topics for the provided messages.

        Args:
            messages (ProcessedMessage | List[ProcessedMessage]): A single ProcessedMessage object or a list of
                                                                   ProcessedMessage objects.

        Returns:
            ProcessedMessage | List[ProcessedMessage]: A single ProcessedMessage object or a list of
                                                       ProcessedMessage objects with topics defined.
        """
        kwargs = {'prompt_func': Prompt.category_prompt,
                  'setup': SETUPS['high']}

        if isinstance(messages, ProcessedMessage):
            return await cls._process_single_message(messages,
                                                     **kwargs)

        elif isinstance(messages, list):
            return await cls._process_multi_message(messages,
                                                    **kwargs
                                                    )
        else:
            raise ValueError('Messages must be a list or a ProcessedMessage')
