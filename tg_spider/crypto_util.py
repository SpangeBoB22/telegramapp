import json
import logging
from pathlib import Path

from Crypto.Cipher import AES
from Crypto.Protocol.KDF import PBKDF2
from Crypto.Random import get_random_bytes

logger = logging.getLogger(__name__)


class JsonEncryptor:
    """
    Class for encrypting and decrypting JSON files.

    Attributes:
        block_size (int): Block size for encryption.
    """

    def __init__(self):
        """Initializes JsonEncryptor with the base block size for AES."""
        self.block_size = AES.block_size

    def encrypt_json(self, file_path: Path, password: str, delete_initial_file: bool = False):
        """
        Encrypts the content of a JSON file, saving the result in a new file with .enc extension.

        Args:
            file_path (Path): Path to the source JSON file.
            password (str): Password for generating the encryption key.
            delete_initial_file (bool): Whether to delete the original file after encryption.

        Notes:
            Example of usage:
            >>>encryptor = JsonEncryptor()
            >>>sample_file_path = Path('path/to/your/file.json')
            >>>sample_password = 'your_secure_password'
            >>>encryptor.encrypt_json(sample_file_path, sample_password, delete_initial_file=True)
        """
        with file_path.open('r', encoding='utf-8') as file:
            data = json.dumps(json.load(file))

        salt = get_random_bytes(16)
        key = PBKDF2(password, salt, dkLen=32, count=1000000)

        cipher = AES.new(key, AES.MODE_CBC)
        ct_bytes = cipher.encrypt(self.pad(data.encode('utf-8')))

        encrypted_file_path = file_path.with_suffix('.enc')
        with encrypted_file_path.open('wb') as file:
            file.write(salt + cipher.iv + ct_bytes)

        if delete_initial_file:
            file_path.unlink()

        logger.info(f"Config file encrypted: {encrypted_file_path}")

    def decrypt_json(self, file_path: Path, password: str):
        """
        Decrypts an .enc file, returning its content as a JSON object.

        Args:
            file_path (Path): Path to the encrypted file.
            password (str): Password used during encryption.

        Notes:
            Example of usage:
            >>>encryptor = JsonEncryptor()
            >>>encrypted_file_path = Path('path/to/your/file.json.enc')
            >>>sample_decrypted_data = encryptor.decrypt_json(encrypted_file_path, password)
            >>>print(sample_decrypted_data)

        Returns:
            dict: Decrypted JSON data.
        """
        with file_path.open('rb') as file:
            encrypted_data = file.read()

        salt = encrypted_data[:16]
        iv = encrypted_data[16:32]
        ct = encrypted_data[32:]
        key = PBKDF2(password, salt, dkLen=32, count=1000000)

        cipher = AES.new(key, AES.MODE_CBC, iv)
        pt = self.unpad(cipher.decrypt(ct)).decode('utf-8')

        decrypted_data = json.loads(pt)
        logger.info('Config file was decrypted into JSON data')
        return decrypted_data

    def pad(self, s):
        """
        Pads input data to a size that is a multiple of the block size.

        Args:
            s (bytes): Input data to pad.

        Returns:
            bytes: Padded data.
        """
        return s + (self.block_size - len(s) % self.block_size) * chr(
            self.block_size - len(s) % self.block_size).encode()

    @staticmethod
    def unpad(s):
        """
        Removes padding from the data.

        Args:
            s (bytes): Padded data.

        Returns:
            bytes: Data without padding.
        """
        return s[:-ord(s[len(s) - 1:])]
