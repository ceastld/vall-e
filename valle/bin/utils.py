import json
import os
from os import PathLike
from pathlib import Path
import random
import shutil
from typing import Callable, List, Tuple
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
    
def copy_file(source_file, destination_folder):
    # 检查源文件是否存在
    if not os.path.isfile(source_file):
        print(f"源文件 '{source_file}' 不存在。")
        return

    # 检查目标文件夹是否存在，如果不存在则创建
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # 获取源文件的文件名
    file_name = os.path.basename(source_file)

    # 构建目标文件的完整路径
    destination_file = os.path.join(destination_folder, file_name)

    try:
        # 复制文件
        shutil.copy2(source_file, destination_file)
        print(f"成功将文件 '{source_file}' 复制到 '{destination_file}'。")
    except Exception as e:
        print(f"复制文件时发生错误：{str(e)}")
        

def create_file_base(
    file_name: PathLike,
    ids: List[str],
    get_info: Callable[[str], Tuple[str, PathLike]],
    texts: List[str],
    infer_dir: PathLike,
    copy: bool = False,
):
    with open(file_name, "w") as file:
        used = {}
        for n, id in enumerate(ids):
            text_prompt, audio_prompt = get_info(id)
            text = texts[n % len(ids)]
            if id in used:
                used[id] += 1
                suffix = f"infer{used[id]}"
            else:
                used[id] = 0
                suffix = "infer"
            audio_out = f"{infer_dir}/{id}_{suffix}.wav"
            file.write(f"{text_prompt}\t{audio_prompt}\t{text}\t{audio_out}\n")
    print(open(file_name).read())
    if copy:
        shutil.copyfile(file_name, f"../../../{file_name}")
    return file_name