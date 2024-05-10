from typing import List, Literal

from numpy._typing import NDArray

from ...core.models import ModelSetup
from . import MODELS


class LLM_Model:

    def __init__(self, setup_model: ModelSetup):
        self._setup = setup_model

    @property
    def name(self):
        return self._setup.name

    @property
    def props(self):
        """

        Model properties as a dict.
        Can be useful for debugging purposes and model's comparison

        """
        response = {'temp': self._setup.temperature}
        if self._setup.top_p is not None:
            response.update({'top_p': self._setup.top_p})

        if self._setup.top_k is not None:
            response.update({'top_k': self._setup.top_k})

        if self._setup.freq_penalty is not None:
            response.update({'freq_penalty': self._setup.freq_penalty})

        if self._setup.pres_penalty is not None:
            response.update({'pres_penalty': self._setup.pres_penalty})

        return response

    @property
    def props_as_str(self):
        """

        Model properties as a string.
        Can be useful for debugging purposes and model's comparison

        """
        strings = [f'{k} = {v}' for k, v in self.props.items()]
        return '; '.join(strings)

    def __enter__(self):
        self._model = MODELS.get(self._setup.name)(self._setup)
        return self

    async def __aenter__(self):
        self._model = MODELS.get(self._setup.name)(self._setup)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._model.close()

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self._model.aclose()

    def answer(self, background: str, question: str) -> str:
        """
        Generates a synchronous response to a question using a fine-tuned model.

        This method sends a chat completion request to the model's API, including a predefined
        background context and the user's question. It then waits for and returns the response.

        Args:
            background (str): The background for the prompts in the actual chat-format
            question (str): The user's question to send to the model.

        Returns:
            str: The content of the model's response as a string.
        """
        return self._model.sync_answer(background, question)

    async def answer_async(self, background: str, question: str) -> str:
        """
        Generates an asynchronous response to a question using a fine-tuned model.

        Similar to `sync_answer`, this method sends a chat completion request to the
        model's API asynchronously. It includes a predefined background context and
        the user's question and returns the response once it's received.

        Args:
            background (str): The background for the prompts in the actual chat-format
            question (str): The user's question to send to the model.

        Returns:
            str: The content of the model's response as a string.
        """
        try:
            return await self._model.async_answer(background, question)
        except Exception as e:
            return f'{self._model.name} failed to answer. Error: {e}'

    async def get_embeddings(self, texts: List[str], kind_of_embedding: Literal['similarity', 'classification'])\
            -> List[NDArray] | str:
        try:
            return await self._model.get_embeddings(texts, kind_of_embedding)
        except Exception as e:
            return f'{self._model.name} failed to get embeddings. Error: {e}'
