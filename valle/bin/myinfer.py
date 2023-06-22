from os import PathLike
import os
import torch
import torchaudio
from valle.data.collation import get_text_token_collater
from valle.data.tokenizer import (
    AudioTokenizer,
    TextTokenizer,
    tokenize_audio,
    tokenize_text,
)

from valle.models.valle import VALLE

class InferModel:
    def __init__(
        self,
        checkpoint: PathLike,
        decoder_dim: int = 1024,
        nhead: int = 16,
        num_decoder_layers: int = 12,
        text_extractor: str = "pypinyin_initials_finals",  # or espeak
        text_tokens_path: PathLike = "data/tokenized/unique_text_tokens.k2symbols",
    ) -> None:
        model = VALLE(
            d_model=decoder_dim,
            nhead=nhead,
            num_layers=num_decoder_layers,
        )
        self.model = model
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = device
        if os.path.isfile(checkpoint):
            checkpoint = torch.load(checkpoint, map_location=device)
            missing_keys, _ = model.load_state_dict(checkpoint["model"], strict=True)
            assert not missing_keys
        model.to(device).eval()
        self.text_tokenizer = TextTokenizer(backend=text_extractor)
        self.text_collater = get_text_token_collater(text_tokens_path)
        self.audio_tokenizer = AudioTokenizer()

    def infer(
        self,
        text_prompt: str,
        audio_prompt_path: PathLike,
        text: str,
        save_path: PathLike,
        top_k: int = -100,
        temperature: float = 1.0,
        continual=False,  # TODO 实现continual 部分
    ):
        print(f"synthesize text: {text}")
        text_tokens, text_tokens_lens = self.text_collater(
            [tokenize_text(self.text_tokenizer, text=f"{text_prompt} {text}".strip())]
        )
        _, enroll_x_lens = self.text_collater(
            [tokenize_text(self.text_tokenizer, text=f"{text_prompt}".strip())]
        )
        device = self.device
        audio_prompt = (
            tokenize_audio(self.audio_tokenizer, audio_prompt_path)[0][0]
            .transpose(2, 1)
            .to(device)
        )
        encoded_frames = self.model.inference(
            text_tokens.to(device),
            text_tokens_lens.to(device),
            audio_prompt,
            enroll_x_lens=enroll_x_lens,
            top_k=top_k,
            temperature=temperature,
        )
        samples = self.audio_tokenizer.decode([(encoded_frames.transpose(2, 1), None)])
        torchaudio.save(save_path, samples[0].cpu(), 24000)
        
    def infer_by_file(self,file_name:PathLike):
        lines = open(file_name).readlines()
        for line in lines:
            org = line.strip()
            if org == "":
                continue
            self.infer(*org.split('\t'))

