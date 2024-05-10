from unittest import TestCase
from app.core.models import ProcessedMessage

jsonl = '{"publish_date": "2024-03-23T12:40:27+00:00", "source": "Березовый сок", "text": "Внезапно... На бис???"}'


class TestModels(TestCase):

    def test_processed_message_from_json(self):
        pm = ProcessedMessage.from_jsonl_dump(jsonl)
        pass



