import asyncio
import pathlib
from typing import Optional, Literal
from pathlib import Path
from colorama import init, Fore, Style

import click
from app.tg_spider.crypto_util import JsonEncryptor
from app.tg_spider.telethone_wrapper import TelegramWorker
from app.core.es_connector import AsyncElasticsearchConnector
from app.process_run import ProcessRunner

init(autoreset=True)


@click.group()
def cli():
    pass


@cli.command()
@click.argument('file_path', type=click.Path(exists=True, dir_okay=False), metavar='path/to/JSON/file')
@click.argument('password', metavar='password')
@click.option('--delete-initial-file', is_flag=True, help="Delete the original file after encryption")
def encrypt_config(file_path: str, password: str, delete_initial_file: bool):
    """
    Encrypts a JSON configuration file.
    """
    file_path = Path(file_path)
    encryptor = JsonEncryptor()
    encryptor.encrypt_json(file_path, password, delete_initial_file=delete_initial_file)
    click.echo(f"Encrypted file saved as {file_path.with_suffix('.enc')}")


@cli.command()
@click.argument('worker_id', type=click.INT, metavar='worker_id')
@click.option('--password', '-p', required=True, type=click.STRING, help='Password for the credential decryption')
@click.option('--output', '-o', default=None, type=click.Path(file_okay=True, dir_okay=False),
              help='Output file path and name')
def list_channels(worker_id: int, password: str, output: Optional[str]):
    """
    Lists channels accessible to a given worker ID after decrypting credentials.
    """
    TelegramWorker.get_channels_list(worker_id, password, output)


@cli.command()
@click.option('--password', '-p', type=click.STRING, required=True,
              help='Password to the credential decryption')
def start_all_readers(password: str):
    """
    Starts reading messages from channels for all workers' ID.
    """
    TelegramWorker.start_reading_all_channels(password)


@cli.command()
@click.argument('worker_id', type=click.INT, metavar='worker_id')
@click.option('--password', '-p', type=click.STRING, required=True,
              help='Password to the credential decryption')
def start_single_reader(worker_id: int, password: str):
    """
    Starts reading messages from channels for a specific worker ID.
    """
    TelegramWorker.start_reading_channels(worker_id, password)


@cli.command()
@click.argument('output_path',
                type=click.Path(exists=False, file_okay=False, dir_okay=True, writable=True),
                metavar='path/to/output/dir')
@click.option('--source', '-s',
              type=str,
              help='Channel name. If absent, the dump will be done for all the sources.')
def dump_index(output_path: str, source: str):
    """
    Dumps the index with messages into the given folder
    """
    # Перетворення шляху в об'єкт Path, якщо output_path не None
    output_path = pathlib.Path(output_path) if output_path else None
    if source is not None:
        click.echo(f"Dumping {source} to {output_path}")
    else:
        click.echo(f"Dumping all channels to {output_path}")
    AsyncElasticsearchConnector().save_es_index_to_files(output_path, source)


@cli.command()
@click.argument('input_path',
                type=click.Path(exists=True, file_okay=False, dir_okay=True),
                metavar='path/to/input/dir')
def load_data(input_path: str):
    """
    Loads the data from the files in the given folder
    """
    input_path = pathlib.Path(input_path)
    if input_path is None:
        click.echo("Input path is empty")

    AsyncElasticsearchConnector().load_from_files(input_path)


@cli.command()
@click.argument('action',
                type=click.Choice(['all', 'clear_and_ner', 'summary', 'tonality', 'topics', 'embeddings']),
                metavar='[clean_and_ner|summary|tonality|topics]')
@click.option('--source', '-s', default=None, help='Name of the news channel')
def llm_process(action: Literal['all', 'clear_and_ner', 'summary', 'tonality', 'topics', 'embeddings'],
                source: str | None):
    """
    Starts a single process for one or all the channels, which are already in ES index.
    """
    pr = ProcessRunner()
    match action:
        case 'all':
            pr.process_all(source)
        case 'clear_and_ner':
            pr.process_clear_and_ner(source)
        case 'summary':
            pr.process_summary(source)
        case 'tonality':
            pr.process_tonality(source)
        case 'topics':
            pr.process_category(source)
        case 'embeddings':
            pr.generate_embeddings(source)
        case _:
            raise ValueError('Illegal action name')


@cli.command()
def process():
    """
    Processes all the unprocessed data in the ES index
    """
    pr = ProcessRunner()
    pr.process_as_daemon()


@cli.command()
def detailed_help():
    """Shows detailed help for all the commands"""
    for name, command in cli.commands.items():
        click.echo(Fore.GREEN + ('\n' + name) + Style.RESET_ALL)
        click.echo(command.get_help(command.context_class(command)))


if __name__ == '__main__':
    cli()
