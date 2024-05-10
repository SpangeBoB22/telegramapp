import datetime
import pathlib
import re
from typing import Generator, Sized

import pytz


def sanitize_string(value: str) -> str:
    """
    Sanitize a filename by replacing invalid characters with underscores.

    This method replaces any invalid characters in the filename with underscores.

    Args:
        value (str): The filename to sanitize.

    Returns:
        str: The sanitized filename.
    """
    return re.sub(r'[\/\\:*?"<>|]', '_', value)


def remove_markdown_and_non_text(text: str, place_space: bool = False) -> str:
    """
    Removes Markdown markup and all non-text characters from the given text.

    Args:
        text (str): The original text from which to remove markup and non-text characters.
        place_space (bool): if True, non text characters will be replaced with a space, otherwise they will be deleted.

    Returns:
        str: Text without markup and non-text characters.
    """

    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    text = re.sub(r'__(.+?)__', r'\1', text)
    text = re.sub(r'[^a-zA-Zа-яА-ЯІіЇїЄєҐ\—\'\/\ґ\s\.,!?0\d\(\)]', ' ' if place_space else '', text)
    text = re.sub(r'\n{2,}', '\n', text)
    text = re.sub(r'\n', ' ', text)
    if place_space:
        text = re.sub(r'\s{2,}', ' ', text)

    return text


def chunker(seq: Sized, size: int) -> Generator:
    """
    Divides a sequence into chunks of the specified size.

    Args:
        seq (sized): The iterable sequence to be divided.
        size (int): The size of each chunk.

    Yields:
        iterable: Chunks of the sequence with the specified size.
    """
    for pos in range(0, len(seq), size):
        yield seq[pos:pos + size]


def seconds_to_hms(seconds: float) -> str:
    """
    Converts the given number of seconds into the HH:MM:SS time format in the Kiev timezone.

    Args:
        seconds (float): The number of seconds to convert into time format.

    Returns:
        str: A string in the HH:MM:SS time format, considering the Kiev timezone.
    """
    kiev_tz = pytz.timezone('Europe/Kiev')
    datetime_utc = datetime.datetime.fromtimestamp(seconds, kiev_tz)
    return datetime_utc.strftime("%H:%M:%S")


class FileGenerator:

    def __init__(self, path: pathlib.Path):
        self._files = list(path.glob('*.jsonl'))
        self.i = 0
        self.stop = len(self._files)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.i >= self.stop:
            raise StopAsyncIteration

        self.i += 1
        return self._files[self.i - 1]
