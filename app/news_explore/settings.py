from ..core.models import ModelSetup, ModelNames

MESSAGES_COUNT_THRESHOLD = {ModelNames.GPT4: 8,
                            ModelNames.GPT3: 8,
                            ModelNames.GEMINI: 6}

N_OF_ASYNC_REQUESTS = {ModelNames.GPT4: 50,
                       ModelNames.GPT3: 50,
                       ModelNames.GEMINI: 12}

SETUPS = {'low': ModelSetup(name=ModelNames.GEMINI,
                            temperature=0.4,
                            top_p=0.95,
                            top_k=40),
          'high': ModelSetup(name=ModelNames.GEMINI,
                             temperature=0.8,
                             top_p=0.8),
          'embeddings': ModelSetup(name=ModelNames.GEMINI)}

EMBEDDINGS_THRESHOLD = {ModelNames.GPT4: 30,
                        ModelNames.GPT3: 30,
                        ModelNames.GEMINI: 30}
