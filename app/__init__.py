import base64
import configparser
import os
import pathlib

from dotenv import load_dotenv


class Config:
    """

    Configuration management class for the application.

    This class is responsible for loading and providing access to configuration settings
    required by the application. It supports reading configuration settings from both a
    .cfg file and environment variables. Environment variables are loaded from a .env file
    specified in the .cfg file. The class provides properties to access specific configuration
    values such as API keys, file paths, and Elasticsearch settings.

    Attributes:
        root_dir (pathlib.Path): The root directory of the project, determined by the location of the file.
        config (configparser.ConfigParser): Loaded configuration from the 'config.cfg' file.

    Properties:
        api_data_file (pathlib.Path): Path to the file containing API credentials.
        channels_list_file (pathlib.Path): Path to the file containing a list of channels.
        es_api_key (tuple): A tuple containing the ID and the secret part of the Elasticsearch API key, decoded from an environment variable.
        es_host (str): The hostname of the Elasticsearch server.
        es_index_name (str): The name of the Elasticsearch index to be used.
        es_port (int): The port number on which the Elasticsearch server is listening.
        es_scheme (str): The scheme used for the Elasticsearch connection ('http' or 'https').
        sessions_root (pathlib.Path): Path to the directory for storing sessions. It's created if it doesn't exist.

    Raises:
        FileNotFoundError: If the .env file specified in the 'config.cfg' cannot be found or loaded.

    Note:
        The .env file should contain sensitive or environment-specific variables such as API keys,
        ensuring they are not hard-coded into the application's source code.

    """
    def __init__(self):
        self.root_dir = pathlib.Path(__file__).parents[1].absolute()
        self.config = configparser.ConfigParser()
        self.config.read(self.root_dir / 'config.cfg')

        if not load_dotenv(self.root_dir / self.config['DEFAULT']['ENV_FILE']):
            raise FileNotFoundError('.env file not found.')

    @property
    def api_data_file(self):
        return self.root_dir / self.config['DEFAULT']['CREDS_FILE']

    @property
    def channels_list_file(self):
        return self.root_dir / self.config['DEFAULT']['CHANNELS_FILE']

    @property
    def days_depth(self):
        try:
            return int(self.config['DEFAULT']['DAYS_DEPTH'])
        except ValueError| TypeError:
            return 0

    @property
    def es_api_key(self):
        encrypted_key = os.getenv('ES_API_KEY', None)
        if not encrypted_key:
            return None, None
        else:
            decoded_key = base64.b64decode(encrypted_key).decode('utf-8')
            return decoded_key.split(':')

    @property
    def es_batch_size(self) -> int:
        return int(self.config['ES']['BATCH_SIZE'])

    @property
    def es_host(self):
        return self.config['ES']['HOST']

    @property
    def es_index_name(self):
        return self.config['ES']['INDEX_NAME']

    @property
    def es_n_connections(self):
        return int(self.config['ES'].get('N_OF_CONNECTIONS', 10))

    @property
    def es_port(self):
        return self.config['ES']['PORT']

    @property
    def es_scheme(self):
        return self.config['ES']['SCHEME']

    @property
    def es_use_batch(self):
        return bool(self.config['ES']['USE_BATCH'])

    @property
    def es_process_chunk(self):
        return int(self.config['ES']['CHUNK'])

    @property
    def sessions_root(self):
        path_to_sessions = self.root_dir / 'sessions'
        if not path_to_sessions.exists():
            path_to_sessions.mkdir()
        return path_to_sessions


conf = Config()
