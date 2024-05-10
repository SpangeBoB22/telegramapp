from unittest import TestCase

from tg_spider import conf


class TestConfig(TestCase):

    def test_api_data_path(self):
        actual = conf.api_data_file
        self.assertEqual('api_data.enc', actual)
