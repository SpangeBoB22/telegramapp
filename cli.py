import asyncio
import pathlib
from typing import Optional
from pathlib import Path

import click
from tg_spider.crypto_util import JsonEncryptor
from tg_spider.telethone_wrapper import TelegramWorker
from tg_spider.es_connector import AsyncElasticsearchConnector


@click.group()
def cli():
    """Defines the CLI group for handling Telegram worker commands."""
    pass


@cli.command()
@click.argument('file_path', type=click.Path(exists=True, dir_okay=False))
@click.argument('password')
@click.option('--delete-initial-file', is_flag=True, help="Delete the original file after encryption")
def encrypt_config(file_path: str, password: str, delete_initial_file: bool):
    """
    Encrypts a JSON configuration file.

    Args:
        file_path (str): Path to the source JSON file.
        password (str): Password for generating the encryption key.
        delete_initial_file (bool): Flag to delete the original file after encryption.

    Encrypts the given JSON configuration file using the specified password and, optionally,
    deletes the original file. Outputs the path of the encrypted file.
    """
    file_path = Path(file_path)
    encryptor = JsonEncryptor()
    encryptor.encrypt_json(file_path, password, delete_initial_file=delete_initial_file)
    click.echo(f"Encrypted file saved as {file_path.with_suffix('.enc')}")


@cli.command()
@click.option('--worker_id', '-w', required=True, type=click.INT, help='Worker ID')
@click.option('--password', '-p', required=True, type=click.STRING, help='Password for the credential decryption')
@click.option('--output', '-o', default=None, type=click.Path(file_okay=True, dir_okay=False), help='Output file path and name')
def list_channels(worker_id: int, password: str, output: Optional[str]):
    """
    Lists channels accessible to a given worker ID after decrypting credentials.

    Args:
        worker_id (int): Worker ID to use for fetching channel list.
        password (str): Password for decrypting the credentials.
        output (Optional[str]): Path to save the output file. If not provided, prints to stdout.
    """
    TelegramWorker.get_channels_list(worker_id, password, output)


@cli.command()
@click.option('--password', '-p', required=True, type=click.STRING, help='Password for the credential decryption')
def start_all_readers(password: str):
    """
    Starts reading messages from channels for all workers' ID.

    Args:
        password (str): Password for decrypting the credentials.
    """
    TelegramWorker.start_reading_all_channels(password)


@cli.command()
@click.option('--worker_id', '-w', required=True, type=click.INT, help='Worker ID')
@click.option('--password', '-p', required=True, type=click.STRING, help='Password for the credential decryption')
def start_single_reader(worker_id: int, password: str):
    """
    Starts reading messages from channels for a specific worker ID.

    Args:
        worker_id (int): Worker ID to use for reading messages.
        password (str): Password for decrypting the credentials.
    """
    TelegramWorker.start_reading_channels(worker_id, password)


@cli.command()
@click.option('--output_path', '-o',
              type=click.Path(exists=False, file_okay=False, dir_okay=True, writable=True),
              help='Path to output dir')
def dump_index(output_path: str):
    """
    Dumps the index with messages into the given folder

    Args:
        output_path (str): The path to the output dir.
    """

    # Перетворення шляху в об'єкт Path, якщо output_path не None
    output_path = pathlib.Path(output_path) if output_path else None

    AsyncElasticsearchConnector().save_es_index_to_files(output_path)


if __name__ == '__main__':
    cli()
