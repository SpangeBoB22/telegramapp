import datetime
import json
from collections import namedtuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, List, Any
import hashlib

import dacite
import numpy as np
from dacite import from_dict

from app.constants import GEMINI_MODEL, AI_4_MODEL, AI_3_MODEL
from app.core.helpers import remove_markdown_and_non_text, sanitize_string

API_data = namedtuple('API_data', 'session_name api_id api_hash')


class ModelNames(Enum):

    ABSTRACT = "abstract_model"
    GEMINI = GEMINI_MODEL
    GPT4 = AI_4_MODEL
    GPT3 = AI_3_MODEL


MODELS_ALIASES = {'gpt4': ModelNames.GPT4,
                  'gpt3': ModelNames.GPT3,
                  'gemini': ModelNames.GEMINI}


@dataclass(frozen=True)
class ChannelMessage:

    """

    Channel message to be saved in the ES cluster
    All the fields are retrieved from the fields of the Telethone Message instance

    """

    message_id: str                     # unique message ID from Telegram API
    publish_date: datetime.datetime     # message date
    message_text: str                   # message text
    source: str                         # the channel name the message is retrieved from

    @property
    def id_(self):
        return self.message_id

    def get_body(self) -> Dict[str, str]:
        return {'publish_date': self.publish_date,
                'source': remove_markdown_and_non_text(self.source, place_space=True),
                'initial_text': self.message_text}


@dataclass
class ModelSetup:
    name: ModelNames | None = None
    temperature: float = 0.3
    top_p: float | None = None
    top_k: int | None = None
    freq_penalty: float | None = None
    pres_penalty: float | None = None
    max_tokens: int | None = 1024
    timeout: int | None = None

    @staticmethod
    def from_dict(data: dict) -> "ModelSetup":
        if 'name' in data:
            try:
                data['name'] = ModelNames(data['name'])
            except ValueError:
                try:
                    data['name'] = ModelNames(MODELS_ALIASES[data['name'].lower()])
                except ValueError:
                    raise ValueError('Unknown model name')

        return dacite.from_dict(ModelSetup, data)

    def __str__(self) -> str:
        return f'{self.name.value} (temp={self.temperature:.2f})'


@dataclass
class ProcessedMessage:
    """

    Represents a processed message to be loaded or saved with the ES cluster

    """

    message_id: str | None
    publish_date: datetime.datetime | None
    source: str | None
    initial_text: str
    cleared_text: str | None = None
    named_entities: List[str] = field(default_factory=list)
    summary: str | None = None
    tonality: str | None = None
    category: str | None = None
    embedding_sts: np.ndarray | None = None
    embedding_clf: np.ndarray | None = None

    @property
    def id_(self) -> str:
        """
        The id of the message
        """
        return self.message_id

    @staticmethod
    def create_from_str(text: str) -> 'ProcessedMessage':
        """
        Create a processed message from a string.
        Used in the tests' mockups
        """
        return ProcessedMessage(message_id=None,
                                publish_date=None,
                                initial_text=remove_markdown_and_non_text(text),
                                source=None)

    @staticmethod
    def from_jsonl_dump(json_line: str) -> 'ProcessedMessage' or None:
        """
        Create a processed message from a jsonl string.
        """
        data = json.loads(json_line)
        if data['text'] is not None:
            if data['text'] != '':
                return ProcessedMessage(message_id=hashlib.md5(data['text'].encode('utf-8')).hexdigest(),
                                        publish_date=datetime.datetime.strptime(data['publish_date'],
                                                                                "%Y-%m-%dT%H:%M:%S%z"),
                                        source=data['source'],
                                        initial_text=remove_markdown_and_non_text(data['text']))
        return None

    def get_body(self) -> Dict[str, Any]:
        """
        Get the body of the message for ES processing
        """
        data = asdict(self)
        data.pop('message_id')
        return data

    @staticmethod
    def from_es_source(id_, source) -> 'ProcessedMessage':
        """
        Create a processed message from an ES source.
        """
        source.update({'message_id': id_})
        source.update({'publish_date': datetime.datetime.strptime(source['publish_date'], "%Y-%m-%dT%H:%M:%S%z")})
        if 'embedding_sts' in source:
            source.update({'embedding_sts': np.array(source['embedding_sts'])})
        if 'embedding_clf' in source:
            source.update({'embedding_clf': np.array(source['embedding_clf'])})
        pm = from_dict(ProcessedMessage, source)
        return pm

    @staticmethod
    def create_from_dict(data: Dict) -> 'ProcessedMessage':
        """
        Create a processed message from a dict.
        """
        return from_dict(ProcessedMessage, data)
