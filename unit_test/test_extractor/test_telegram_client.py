from unittest import TestCase
from dotenv import load_dotenv
import pathlib

from tg_spider.telethone_wrapper import TelegramWorker


class TestTelegramClient(TestCase):

    def test_config_getter(self):
        pass

    def test_lister(self):
        TelegramWorker.get_channels_list(1, 'easy_password_123')
