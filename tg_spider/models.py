import datetime
from collections import namedtuple
from dataclasses import dataclass
from typing import Dict

API_data = namedtuple('API_data', 'session_name api_id api_hash')


@dataclass(frozen=True)
class ChannelMessage:

    """

    Channel message to be saved in the ES cluster
    All the fields are retrieved from the fields of the Telethone Message instance

    """

    message_id: str                     # unique message ID from Telegram API
    message_date: datetime.datetime     # message date
    message_text: str                   # message text
    source: str                         # the channel name the message is retrieved from

    @property
    def id_(self):
        return self.message_id

    def get_body(self) -> Dict[str, str]:
        return {'publish_date': self.message_date,
                'source': self.source,
                'text': self.message_text}
