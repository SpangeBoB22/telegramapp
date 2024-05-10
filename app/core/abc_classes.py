from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import fields
from typing import List, Dict, Any

from app.core.models import ModelSetup, ModelNames


class AIModel(ABC):

    _name: ModelNames = ModelNames.ABSTRACT
    _allowed_setup = List[str]

    def __init__(self, setup: ModelSetup) -> None:
        self._setup = setup

    @property
    def name(self):
        """

        Model name from the defined list of names

        """
        return self._setup.name.value

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

    @abstractmethod
    def sync_answer(self, background: str, question: str) -> str:
        """

        Generate the answer for the given prompt synchronously

        """
        pass

    @abstractmethod
    async def async_answer(self, background: str, question: str) -> str:
        """

         Generate the answer for the given prompt asynchronously

        """
        pass

    @abstractmethod
    def close(self):
        """
        Closes the connection to the API
        """
        pass

    @abstractmethod
    async def aclose(self):
        """
        Asynchronously closes the connection to the API
        """
        pass

    def _get_kwargs(self) -> Dict[str, Any]:
        """
        Gets kwargs of the model as a dict
        """
        kwargs = defaultdict(None)

        if 'model' in self._allowed_setup:
            kwargs['model'] = self._setup.name.value

        for field in fields(self._setup):
            if field.name not in self._allowed_setup:
                continue
            if (value := getattr(self._setup, field.name)) is not None:
                kwargs[field.name] = value

        return dict(kwargs)
