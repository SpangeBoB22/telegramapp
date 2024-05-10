import asyncio
import json
import pathlib
from itertools import product
from random import sample, seed
from unittest import TestCase

from tqdm.asyncio import tqdm

from news_explore.core import ModelSetup, ModelNames
from news_explore.helpers import remove_markdown_and_non_text
from news_explore.llms.llm_model import LLM_Model
from news_explore.prompts import Prompt

seed(102)
path_to_data = pathlib.Path(__file__).parents[2].resolve() / 'data'
file = sample(list(path_to_data.glob('*.jsonl')), k=1)[0]
with open(file, encoding='utf-8') as f:
    data = f.readlines()

background = 'Ви є помічником, що допомагає людині опрацьовуючі великий обсяг новиних повідомлень'


class TestTextAndSummaryRetrieval(TestCase):

    def setUp(self):
        self._test_data = data[:7]
        self._single_message = json.loads(data[0])['text']
        self._cleaned_messages = list(map(lambda x: remove_markdown_and_non_text(json.loads(x)['text']),
                                          self._test_data))
        self._setup = [ModelSetup(name=ModelNames.GEMINI,
                                  temperature=0.4),
                       ModelSetup(name=ModelNames.GEMINI,
                                  temperature=0.7),
                       ModelSetup(name=ModelNames.GEMINI,
                                  temperature=1)]
        self._low_setup = [ModelSetup(name=ModelNames.GEMINI,
                                      temperature=0.0),
                           ModelSetup(name=ModelNames.GEMINI,
                                      temperature=0.2),
                           ModelSetup(name=ModelNames.GEMINI,
                                      temperature=0.3),
                           ModelSetup(name=ModelNames.GEMINI,
                                      temperature=0.4),
                           ModelSetup(name=ModelNames.GEMINI,
                                      temperature=0.5)
                           ]

    @staticmethod
    async def run_retrieval(model, prompt):
        async with LLM_Model(model) as llm:
            answer = await llm.answer_async(background=background, question=prompt)
        return answer

    @staticmethod
    async def retrieve_async(tasks):
        return await tqdm.gather(*tasks) if tasks else []

    def test_retrieval(self):
        cleaned_message = remove_markdown_and_non_text(self._single_message)
        print(cleaned_message, '\n')

        tasks = [self.run_retrieval(model, prompt_func(cleaned_message))
                 for model, prompt_func in product(self._setup,
                                                   [Prompt.extract_prompt,
                                                    Prompt.summary_prompt,
                                                    Prompt.named_entities])]
        details = [(str(model), prompt_func.__name__)
                   for model, prompt_func in product(self._setup,
                                                     [Prompt.extract_prompt,
                                                      Prompt.summary_prompt,
                                                      Prompt.named_entities])]

        response = asyncio.run(self.retrieve_async(tasks))

        for detail, item in zip(details, response):
            print(detail, item)

    def test_compare_summary(self):
        cleaned_message = remove_markdown_and_non_text(self._single_message)
        print(cleaned_message, '\n')

        tasks = [self.run_retrieval(model, Prompt.summary_prompt(cleaned_message))
                 for model in self._setup]

        response = asyncio.run(self.retrieve_async(tasks))

        for detail, item in zip(self._setup, response):
            print(str(detail), '\t', item)


    def test_message_formatter(self):
        for n, message in enumerate(self._cleaned_messages):
            print(f'{n + 1}. {message}')

        tasks = [self.run_retrieval(model, Prompt.extract_prompt(message))
                 for model, message in product(self._setup, self._cleaned_messages)]

        response = asyncio.run(self.retrieve_async(tasks))

        for detail, item in zip(product(self._setup, self._cleaned_messages), response):
            print(str(detail[0]), '\t', item)

    def test_low_temp_formatter(self):

        for n, message in enumerate(self._cleaned_messages):
            print(f'{n + 1}. {message}')

        tasks = [self.run_retrieval(model, Prompt.low_temp_prompt(self._cleaned_messages))
                 for model in self._low_setup]

        response = asyncio.run(self.retrieve_async(tasks))

        for detail, item in zip(self._low_setup, response):
            print(str(detail), '\t', item)
