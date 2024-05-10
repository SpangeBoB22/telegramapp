import asyncio

from news_explore.llms.llm_model import LLM_Model
from news_explore.core import ModelSetup, ModelNames
from unittest import TestCase
from concurrent.futures import ThreadPoolExecutor

background = 'Ви є універсальним помічником та відповідаєте на різноманітні питання'
question = 'Розкажи про себе в двох реченнях'


class TestLLmModels(TestCase):

    def setUp(self):
        self.models_setups = {ModelNames.GPT4: ModelSetup(ModelNames.GPT4,
                                                          temperature=0.4),
                              ModelNames.GEMINI: ModelSetup(ModelNames.GEMINI,
                                                            temperature=0.5)}

    @staticmethod
    def run_model(setup_model):
        with LLM_Model(setup_model=setup_model) as model:
            response = model.answer(background, question)
        return response

    @staticmethod
    async def run_model_async(setup_model):
        async with LLM_Model(setup_model=setup_model) as model:
            response = await model.answer_async(background, question)
        return response

    @classmethod
    async def run_retrieval(cls, setups):
        tasks = [cls.run_model_async(model_setup) for model_setup in setups]
        response = await asyncio.gather(*tasks)
        return response

    def test_sync_answer(self):

        with ThreadPoolExecutor() as executor:
            responses = list(executor.map(
                self.run_model, [self.models_setups[ModelNames.GPT4],
                                 self.models_setups[ModelNames.GEMINI]]))

        responses = [f"{model.name.value}: {future}" for future, model in zip(responses,
                                                                              [self.models_setups[ModelNames.GPT4],
                                                                               self.models_setups[ModelNames.GEMINI]])]

        for response in responses:
            print(response)
            self.assertLess(100, len(response))

        self.assertEqual(2, len(responses))

    def test_async_answer(self):
        responses = asyncio.run(self.run_retrieval([self.models_setups[ModelNames.GPT4],
                                                    self.models_setups[ModelNames.GEMINI]]))

        responses = [f"{model.name.value}: {future}" for future, model in zip(responses,
                                                                              [self.models_setups[ModelNames.GPT4],
                                                                               self.models_setups[ModelNames.GEMINI]])]

        for response in responses:
            print(response)

        for response in responses:
            print(response)
            self.assertLess(100, len(response))

        self.assertEqual(2, len(responses))
