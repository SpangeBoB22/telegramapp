import os
from typing import Literal, List

import google.generativeai as genai
import numpy as np
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from numpy._typing import NDArray

from ...core.models import ModelSetup
from ...core.abc_classes import AIModel


class GeminiConnector(AIModel):
    """
    A connector class for Gemini's API to provide synchronous and asynchronous answers,
    and to generate embeddings based on the provided text.

    This class uses both synchronous and asynchronous OpenAI clients to interact with
    the OpenAI API using a specified API key. It supports generating chat completions
    and embeddings with a fine-tuned model and a specific embeddings model.

    """

    _allowed_setup = ['temperature', 'top_p', 'top_k']

    def __init__(self, model_setup: ModelSetup) -> None:

        super().__init__(model_setup)

        genai.configure(api_key=os.environ.get('GEMINI_API_KEY', None))
        kwargs = self._get_kwargs()

        self._generation_config = genai.types.GenerationConfig(
            candidate_count=1,
            stop_sequences=['xxx'],
            max_output_tokens=8192,
            **kwargs)

        self._safety_config = {}
        for value in HarmCategory:
            if value < 7:
                continue
            self._safety_config.update({value: HarmBlockThreshold.BLOCK_NONE})

        self._model = genai.GenerativeModel(model_setup.name.value)

    def sync_answer(self, background: str, question: str) -> str:
        """
        Generates a synchronous response to a question using a fine-tuned model.

        This method sends a chat completion request to the OpenAI API, including a predefined
        background context and the user's question. It then waits for and returns the response.

        Args:
            background (str): The background for the prompts in the actual chat-format
            question (str): The user's question to send to the model.

        Returns:
            str: The content of the model's response as a string.
        """
        response = self._model.generate_content(
            question,
            generation_config=self._generation_config,
            safety_settings=self._safety_config)
        try:
            return response.text
        except Exception as ex:
            # in some cases the Gemini API doesn't return text
            # the cases should be studied and handled properly
            print(ex)
            return ''

    async def async_answer(self, background: str, question: str) -> str:
        """
        Generates an asynchronous response to a question using a fine-tuned model.

        Similar to `sync_answer`, this method sends a chat completion request to the
        OpenAI API asynchronously. It includes a predefined background context and
        the user's question and returns the response once it's received.

        Args:
            background (str): The background for the prompts in the actual chat-format
            question (str): The user's question to send to the model.

        Returns:
            str: The content of the model's response as a string.
        """
        response = None
        try:
            response = await self._model.generate_content_async(
                question,
                generation_config=self._generation_config,
                safety_settings=self._safety_config)
            return response.text
        except Exception as ex:
            # in some cases the Gemini API doesn't return text
            # the cases should be studied and handled properly
            print(ex)
            print(response)
            return "Sorry, no answer was found."

    @staticmethod
    async def get_embeddings(texts: List[str],
                             kind_of_embedding: Literal['similarity', 'classification']) -> List[NDArray] | str:
        response = None
        try:
            response = await genai.embed_content_async(
                model="models/text-embedding-004",
                content=texts,
                task_type="semantic_similarity" if kind_of_embedding == "similarity" else "classification"
            )
            return [np.array(item) for item in response['embedding']]
        except Exception as ex:
            print(ex)
            print(response)
            return "Sorry, no answer was found."

    def close(self):
        """
        Closes the client
        """
        pass

    async def aclose(self):
        """
        Closes the client asynchronously
        """
        pass

