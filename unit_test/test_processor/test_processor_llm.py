import asyncio
import json
import pathlib
from typing import List
from unittest import TestCase
from app.core.models import ProcessedMessage
from app.news_explore.processor_llm import ProcessorLLM

path_to_data = pathlib.Path(__file__).parents[2].resolve() / 'data'
file = list(path_to_data.glob('*.jsonl'))[9]
with open(file, encoding='utf-8') as f:
    data = f.readlines()


class TestProcessorLLM(TestCase):

    def setUp(self):
        self._messages = [ProcessedMessage.create_from_str(text=json.loads(text)['text']) for text in data]
        self.sample_response = ["""
                                |||Текст #1
                                Дія 1: Ворог атакував нашу область і шахедами, і ракетами. Є влучання по обєкту критичної інфраструктури. За попередньою інформацією, постраждала одна людина. Є пошкодження. Наразі перебоїв з електропостачанням в області не зафіксовано. На місці влучання працюють рятувальники ДСНС. Більше інформації згодом.
                                Дія 2: Світлана Онищук; ДСНС
                                
                                |||Текст #2
                                Дія 1: ДніпроГЕС стоїть. Раніше в мережі ширилась інформація про влучання росіян в об'єкт.
                                Дія 2: ДніпроГЕС; Андрій Коваленко
                                """,
                                """
                                |||Текст #1
                                Дія 1: Вся Україна — повітряна тривога.
                                Дія 2: 
                                
                                |||Текст #2
                                Дія 1: Монітори повідомляють про зліт МіГ31К. Цей винищувач є потенційним носієм гіперзвукових ракет Х47М2 Кинджал, які не збиваються у більшості регіонів. UPD. Повітряні сили підтверджують інформацію.
                                Дія 2: МіГ31К; Х47М2 Кинджал
                                """,
                                """
                                |||Текст #1
                                Дія 1: Збитки від війни для енергетичної інфраструктури України до сьогоднішньої масованої атаки росіян становили 10,6 млрд.
                                Аналітики UA War Infographics спільно з компанією Yasno порахували збитки для кожної області окремо і підбили загальний підсумок. Водночас, результати дослідження не враховують сьогоднішній масований обстріл, внаслідок якого було пошкоджено десятки енергетичних об'єктів.
                                Дія 2: UA War Infographics; Yasno
                                
                                |||Текст #2
                                Дія 1: По ДніпроГЕС було два прямих влучання. Невідомо, чи вдасться відновити ГЕС2, вона пошкоджена серйозно, — директор Укргідроенерго Ігор Сирота.
                                Є влучання в одну опору, розбиті підкранові балки.
                                Доведеться відновлювати повністю машинний зал і електричне обладнання. Наслідки ми прорахуємо протягом дня і будемо розуміти, що відбулося. І чи зможе вона (ГЕС2 — ред.) працювати. І чи зможе, то чи в обмеженому режимі, чи взагалі не зможе певний період, — сказав Сирота в ефірі Радіо Свобода.
                                Дія 2: ДніпроГЕС; ГЕС2; Укргідроенерго; Ігор Сирота; Радіо Свобода
                                """]

    def test_chunk(self):
        messages = self._messages[:6]
        response = asyncio.run(ProcessorLLM.get_clear_text_and_named_entities(messages))
        pass

    def test_postprocessing(self):
        processed = ProcessorLLM._post_process_messages_clean_and_ner(self._messages[:6], self.sample_response)
        pass

    def test_summary(self):
        messages = self._messages[:6]
        response = ProcessorLLM.get_summary(messages)
        pass

    def test_sequential_process(self):
        messages = self._messages[:6]
        response = asyncio.run(self._sequential_process(messages))
        pass

    @staticmethod
    async def _sequential_process(messages) -> List[ProcessedMessage]:
        cleared_messages = await ProcessorLLM.get_clear_text_and_named_entities(messages)
        return await ProcessorLLM.get_summary(cleared_messages)

