import json
from unittest import TestCase
from pathlib import Path
import os

from tg_spider.crypto_util import JsonEncryptor


class TestCryptoUtil(TestCase):

    def setUp(self) -> None:
        self.path_to_files = Path(__file__).parent.absolute() / 'files/api_data.json'
        self.path_to_enc = Path(__file__).parent.absolute() / 'files/api_data.enc'
        self.password = 'easy_password_123'

    def test_encrypt(self) -> None:
        JsonEncryptor().encrypt_json(self.path_to_files, password=self.password)

    def test_decrypt(self) -> None:
        actual = JsonEncryptor().decrypt_json(self.path_to_enc, self.password)
        with open(self.path_to_files, 'r') as f:
            expected = json.load(f)
        self.assertDictEqual(expected, actual)
