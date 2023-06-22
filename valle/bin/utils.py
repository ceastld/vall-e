import json
import os
from os import PathLike
from pathlib import Path
import random
from IPython.display import display, Audio
import re


def get_tts_dict():
    """
    从 "tts.json" 文件中读取并返回字典数据

    Returns:
        dict: 包含从 "tts.json" 文件中读取的数据的字典
    """
    module_path = os.path.dirname(os.path.abspath(__file__))
    
    return json.load(open(f"{module_path}/tts.json"))

def get_tts_texts(
    text_type: str = "short",
    count: int = 5,
):
    data = get_tts_dict()
    texts = data[text_type]
    if count < len(texts):
        return random.sample(texts, count)
    else:
        return texts


def random_samples(lst, count: int):
    if count < len(lst):
        return random.sample(lst, count)
    else:
        return lst


def show_audios(text_file):
    with open(text_file) as file:
        for n, line in enumerate(file.readlines()):
            text_prompt, audio_prompt, text, audio_file = line.strip().split("\t")
            print(f"------------{Path(audio_prompt).stem}---------------")
            display(text_prompt, Audio(audio_prompt), text)
            if os.path.exists(audio_file):
                print(audio_file)
                display(Audio(audio_file))


class TextProcessor:
    def __init__(self):
        self.replace_dict = {
            "0": "零",
            "1": "一",
            "2": "二",
            "3": "三",
            "4": "四",
            "5": "五",
            "6": "六",
            "7": "七",
            "8": "八",
            "9": "九",
            "？": "?",
            "！": "!",
            "；": ";",
            "，": ",",
            "。": ".",
            "、": ",",
            "：": ":",
        }
        self.no_replace = re.compile(r"[\u4e00-\u9fff?!;:,.]")
        self.q_marks = re.compile(r"[「」《》]")

    def process_char(self, character):
        if self.q_marks.match(character):
            return '"'
        if self.no_replace.match(character):
            return character
        if character in self.replace_dict:
            return self.replace_dict[character]
        return " "

    def text_pre_process(self, text: str) -> str:
        return "".join([self.process_char(c) for c in text]).strip()
    
