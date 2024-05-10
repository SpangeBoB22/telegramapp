from abc import ABC
from typing import List, Literal

from numpy._typing import NDArray
from openai import AsyncOpenAI, OpenAI

from app.core.abc_classes import AIModel
from app.core.models import ModelSetup, ModelNames


class OpenAIConnector(AIModel, ABC):
    """
    A connector class for OpenAI's API to provide synchronous and asynchronous answers,
    and to generate embeddings based on the provided text.

    This class uses both synchronous and asynchronous OpenAI clients to interact with
    the OpenAI API using a specified API key. It supports generating chat completions
    and embeddings with a fine-tuned model and a specific embeddings model.

    """

    _allowed_setup = ['model', 'temperature', 'top_p', 'frequency_penalty', 'presence_penalty']

    def __init__(self,
                 setup: ModelSetup | None = None):
        super().__init__(setup)
        self._client = OpenAI()
        self._aclient = AsyncOpenAI()

    def sync_answer(self,
                    background: str,
                    question: str):
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
        kwargs = self._get_kwargs()
        completion = self._client.chat.completions.create(
            messages=[
                {"role": "system", "content": background},
                {"role": "user", "content": question}
            ],
            **kwargs
        )

        return completion.choices[0].message.content

    def close(self):
        """
        Closes the client
        """
        self._client.close()

    async def async_answer(self,
                           background: str,
                           question: str):
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
            kwargs = self._get_kwargs()
            completion = self._aclient.chat.completions.create(
                messages=[
                    {"role": "system", "content": background},
                    {"role": "user", "content": question}
                ],
                **kwargs

            )
            response = await completion
            return response.choices[0].message.content
        except Exception as e:
            print(e)
            print(response)
            return "Sorry, no answer was found."

    async def aclose(self):
        """
        Closes the client asynchronously
        """
        await self._aclient.close()

    @staticmethod
    def get_easy_setup():
        """
        Returns the easy setup for the Open AI model.
        Is used for testing purposes.
        """
        return ModelSetup(name=ModelNames.GPT4,
                          temperature=0.5,
                          max_tokens=256
                          )

    async def get_embeddings(self, texts: List[str], kind_of_embedding: Literal['similarity', 'classification']) \
            -> List[NDArray] | str:
        raise NotImplementedError


