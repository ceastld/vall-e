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
            print(f"-------{n}-----{Path(audio_prompt).stem}---------------")
            display(text_prompt, Audio(audio_prompt), text)
            if os.path.exists(audio_file):
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

    def process_str(self, text: str) -> str:
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
    return destination_file
        

def create_file_base(
    file_name: PathLike,
    ids: List[str],
    get_info: Callable[[str], Tuple[str, PathLike]],
    texts: List[str],
    infer_dir: PathLike,
    copy: bool = False,
):
    os.makedirs(infer_dir, exist_ok=True)
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

def copy_file_in_line(line: str, target_path: PathLike):
    """创建用于demo

    Args:
        line (str): 文件中的一行
        target_path (PathLike): 复制到的目标文件夹路径

    Returns:
        _type_: 将文件路径替换过后的一行
    """
    audio_in = line.split("\t")[1]
    audio_out = line.split("\t")[3].strip()
    audio_in_new = copy_file(audio_in, target_path)
    audio_out_new = copy_file(audio_out, target_path)
    return line.replace(audio_in, audio_in_new).replace(audio_out,audio_out_new)

import gzip
import json

def read_jsonl_gz(file_path):
    with gzip.open(file_path, 'rt') as f:
        for line in f:
            yield json.loads(line)

def get_jsonl_gz_data(file_path):
    # 逐行读取和解析 JSON 数据
    json_data = list(read_jsonl_gz(file_path))
    print(json_data)
    
# file_path = './data/manifests/genshin_supervisions_dev.jsonl.gz'

