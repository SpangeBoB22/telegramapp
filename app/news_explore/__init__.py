import pathlib
from dotenv import load_dotenv
from ..core.models import ModelNames

path_to_env = pathlib.Path(__file__).parents[1].absolute() / '.env'
load_dotenv(str(path_to_env))


