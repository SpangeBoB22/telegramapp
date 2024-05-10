from typing import List


class Prompt:

    """
    Creates prompts to the model's APIs
    """

    @staticmethod
    def extract_prompt(text: str) -> str:
        return f"""
        Нижче надано текст є інформаційним повідомленням Телеграм-каналу. З наданого початкового тексту виділіть те,
        що не стосується безпосередньо інформаційного повідомлення, в тому числі ОБОВ'ЯЗКОВО:
            - інформацію про джерело зображень, інфографіки тощо;
            - посилання на чати, канали, групи, боти;
            - хештеги (#) та згадки (@);
            - номери телефонів, посилання на веб-сайти, адреси електронної пошти та інша особиста інформацію.
            - емодзі та інші спеціальні символи.
        У відповіді перелічіть ці елементи через ";" не змінюючи ці елементи. 
        \nПочатковий текст повідомлення: {text}"""

    @staticmethod
    def summary_prompt(text: str) -> str:
        return f"""
        Зроби самиризацію тексту, який наданий нижче, перетворивши його на 1 або два оповідних речення.
        ЗАБОРОНЕНО видаляти іменовані сутності: прізвища, імена та по-батькові діючих осіб, географічні назви,
        власні назви тощо, якщо ці іменовані сутності відносяться до інформаційного повідомлення.
        Відповідь має містити виключно текст самаризації.\nТекст повідомлення: {text}"""

    @staticmethod
    def named_entities(text: str) -> str:
        return f"""
        Знайди у тексті повідомлення, що надане далі, усі іменовані сутності та переліч їх через кому.
        Нічого іншого у відповідь включати не треба.
        \nТекст повідомлення: {text}"""

    @staticmethod
    def low_temp_prompt(texts: List[str]) -> str:
        combined_texts = '\n'.join([f'Текст #{n + 1}: {text}\n' for n, text in enumerate(texts)])
        return f"""
        Нижче надано {len(texts)} текстів, що є повідомленням Телеграм-каналу. Цы повідомлення отримано з каналів новин.
        Отже, тексти містять інформаційне повідомлення та найчастіше певну кількість додаткової інформації.
        З кожним текстом треба зробити наступні дії:
        Дія 1. Рухаючись по нижченаведеному маркованому переліку крок за кроком, видалити з тексту наступні елементи:
             - інформацію про джерело зображень, інфографіки тощо;
             - посилання на чати, канали, групи, боти, інтернет-сторінки разом з текстовими поясненнями до них;
             - посилання на інші повідомлення, повідомлення інших каналів, статті інтернет видань;
             - анонси матеріалів в мережі (статті, вебсторінки, інформаційні повідомлення) ЗА ВИНЯТКОМ тих, що містять
               детальний перелік або опис цих матеріалів більш ніж одним реченням;
             - хештеги (#) та згадки (@);
             - номери телефонів, адреси електронної пошти та іншу особисту інформацію;
             - емодзі та інші спеціальні символи;
             - інші інформацію, що не є безпосередньо частиною центрального інформаційного повідомлення.
           При виконанні видалення елементів з поданого переліку, треба взяти до уваги:       
           - забороняється видаляти заголовки інформаційних повідомлень, що містяться у тексті, у тому числі якщо вони
           сформульовані у якості питання;
           - іноді 
           - 
           Після видалення частин тексту, що не відносяться до інформаційного повідомлення, треба прибрати усі розділові
           знаки окрім наступних: знак питання, знак оклику, крапка, кома, дефіс, тире.
           До відповіді слід додати виключено очищене таким чином повідомлення. У випадку, коли інформаційне 
           повідомлення виділити з наданого тексту неможливо, треба повернути "повідомлення немає".
        2. Знайдіть у тексті повідомлення усі іменовані сутності, у тому числі латиною, та перелічити їх через ";".
           ЗАБОРОНЕНО використовувати лапки та інші розділові знаки окрім дефісу, коми та крапки з комою.
        Результати обробки треба надати послідовно, форматуючи результати обробки кожного окремого наданого тексту
        СУВОРО ДОТРИМУЮЧИСЬ шаблона, оскільки інакше їх неможливо використати:
        |||Текст #<<N>>
        Дія 1: <<відповідь до дії 1 з списку вище>>
        Дія 2: <<відповідь до дії 2 з списку вище>>\n
        Тексти повідомлень:\n{combined_texts}
        """

    @staticmethod
    def summary_chunk_prompt(texts: List[str]) -> str:
        combined_texts = '\n'.join([f'Текст #{n + 1}: {text}\n' for n, text in enumerate(texts)])
        return f"""
        Нижче надано {len(texts)} текстів, що є повідомленням Телеграм-каналу. Ці повідомлення отримано з каналів новин.
        Отже, тексти містять інформаційне повідомлення та найчастіше певну кількість додаткової інформації.
        Зроби самиризацію текстів, які надані нижче, перетворивши кожний з них на одне або два оповідних речення.
        ЗАБОРОНЕНО видаляти іменовані сутності: прізвища, імена та по-батькові діючих осіб, географічні назви, власні
        назви тощо, якщо ці іменовані сутності відносяться до інформаційного повідомлення.
        Результати обробки треба надати послідовно, форматуючи відповідь до кожного тексту в запиті за шаблоном:
        Текст <<N>>: <<результат самаризації тексту N>>
        Якщо самарі тексту відсутнє, у шаблон треба додати "самарі відсутнє". Кількість рядків за шаблоном у відповіді
        ОБОВ'ЯЗКОВО має дорівнювати кількості наданих текстів.
        Тексти повідомлень:\n{combined_texts}
        """

    @staticmethod
    def tonality_prompt(texts: List[str]) -> str:
        combined_texts = '\n'.join([f'Текст #{n + 1}: {text}\n' for n, text in enumerate(texts)])
        return f"""
        Нижче надано {len(texts)} текстів, що є повідомленням Телеграм-каналу. Ці повідомлення отримано з каналів новин.
        Отже, тексти містять інформаційне повідомлення та найчастіше певну кількість додаткової інформації.
        Вважатимемо, що кожне інформаційне повідомлення має одну з наступних тональностей: позитивна, негативна,
        нейтральна, об'єктивна, емоційна, загрозлива. Всього шість можливих тональностей.
        Для кожного з текстів, які надані нижче, визнач тональність, обравши найближчу за сенсом з наданного вище
        переліку можливих тональностей повідомлення. Тональність має бути призначена кожному повідомленню обов'язково.
        Результати обробки треба надати послідовно, форматуючи відповідь до кожного тексту в запиті за шаблоном:
        Текст <<N>>: <<тональність тексту N>>
        Кількість рядків за шаблоном у відповіді ОБОВ'ЯЗКОВО має дорівнювати кількості наданих текстів.
        Тексти повідомлень:\n{combined_texts}
        """

    @staticmethod
    def category_prompt(texts: List[str]) -> str:
        combined_texts = '\n'.join([f'Текст #{n + 1}: {text}\n' for n, text in enumerate(texts)])
        return f"""
        Нижче надано {len(texts)} текстів, що є повідомленням Телеграм-каналу. Ці повідомлення отримано з каналів новин.
        Отже, тексти містять інформаційне повідомлення та найчастіше певну кількість додаткової інформації.
        Вважатимемо, що кожне інформаційне повідомлення має одну з наступних категорій: політика, економіка,
        соціальні питання, наука і технології, культура і мистецтво, спорт, міжнародні події, освіта,
        здоров'я і медицина, природні катастрофи та надзвичайні ситуації. Всього 10 категорій.
        Для кожного з текстів, які надані нижче, визнач категорію, обравши найближчу за сенсом з наданного вище
        переліку можливих категорій повідомлення. Категорія має бути призначена кожному повідомленню обов'язково.
        Результати обробки треба надати послідовно, форматуючи відповідь до кожного тексту в запиті за шаблоном:
        Текст <<N>>: <<категорія тексту N>>
        Кількість рядків за шаблоном у відповіді ОБОВ'ЯЗКОВО має дорівнювати кількості наданих текстів.
        Тексти повідомлень:\n{combined_texts}
        """
