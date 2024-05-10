import logging
from logging.handlers import TimedRotatingFileHandler
import traceback

from app.core import Singleton


class AsyncLogger(metaclass=Singleton):

    def __init__(self):
        self._logger = logging.getLogger('tg_spider')
        self._logger.setLevel(logging.DEBUG)
        self._configure_logger()

    def _configure_logger(self):
        handler = TimedRotatingFileHandler("tg_app.log", when="D", interval=1, backupCount=7, encoding="utf-8")
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self._logger.addHandler(handler)

    async def alog(self, level: int, message: str, ex: Exception | None = None):
        tb_str = ''
        if ex is not None:
            tb_exception = traceback.TracebackException.from_exception(ex)
            tb_str = ''.join(tb_exception.format())
        self._logger.log(level, f'{message}\nError: {ex}\n{tb_str}' if ex is not None else message)

    def log(self, level: int, message: str, ex: Exception | None = None):
        tb_str = ''
        if ex is not None:
            tb_exception = traceback.TracebackException.from_exception(ex)
            tb_str = ''.join(tb_exception.format())
        self._logger.log(level, f'{message}\nError: {ex}\n{tb_str}' if ex is not None else message)
