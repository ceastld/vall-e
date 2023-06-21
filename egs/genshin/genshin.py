from concurrent.futures import ProcessPoolExecutor
from functools import partial
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional, Union

from tqdm.auto import tqdm

from lhotse import validate_recordings_and_supervisions
from lhotse.audio import Recording, RecordingSet
from lhotse.supervision import SupervisionSegment, SupervisionSet
from lhotse.utils import Pathlike
import json
import argparse

from typing import Dict, Optional, Union


def get_recordings(
    corpus_dir: Pathlike,
    voice: Dict[str, dict],
    num_jobs: int = 1,
) -> RecordingSet:
    msg = "Scanning audio files"

    def get_recording(key):
        return Recording.from_file(
            corpus_dir / voice[key]["fileName"],
            recording_id=key,
        )

    return RecordingSet.from_recordings(
        tqdm(
            map(get_recording, voice.keys()),
            desc=msg,
            total=len(voice),
        )
    )


def prepare_genshin(
    corpus_dir: Pathlike,
    output_dir: Optional[Pathlike],
    num_jobs: int = 1,
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]:
    """
    Returns the manifests which consist of the Recordings and Supervisions
    :param corpus_dir: Pathlike, the path of the data dir.
    :param output_dir: Pathlike, the path where to write the manifests.
    :return: a Dict whose key is the dataset part,
             and the value is Dicts with the keys 'recordings' and 'supervisions'.
    """
    corpus_dir = Path(corpus_dir)
    assert corpus_dir.is_dir(), f"No such directory: {corpus_dir}"
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    manifests = defaultdict(dict)
    dateset_parts = ["train", "dev", "test"]

    for part in tqdm(
        dateset_parts,
        desc="Process genshin audio",
    ):
        transcript_path = corpus_dir / f"result_chs_{part}.json"
        voice = json.load(open(transcript_path))

        recording_set = get_recordings(
            corpus_dir=corpus_dir,
            voice=voice,
            num_jobs=num_jobs,
        )

        supervisions = []
        for key, value in voice.items():
            npc_name = value["npcName"]
            text = value["text"]

            supervision = SupervisionSegment(
                id=key,
                recording_id=key,
                start=0.0,
                duration=recording_set.duration(key),
                channel=0,
                language="Chinese",
                speaker=npc_name,
                text=text.strip(),
            )
            supervisions.append(supervision)

        supervision_set = SupervisionSet.from_segments(supervisions)
        validate_recordings_and_supervisions(recording_set, supervision_set)

        if output_dir is not None:
            supervision_set.to_file(
                output_dir / f"genshin_supervisions_{part}.jsonl.gz"
            )
            recording_set.to_file(output_dir / f"genshin_recordings_{part}.jsonl.gz")

        manifests[part] = {"recordings": recording_set, "supervisions": supervision_set}
    return manifests


def main():
    parser = argparse.ArgumentParser(description="Prepare Genshin dataset")
    parser.add_argument("--input-dir", type=str, help="Path to the corpus directory")
    parser.add_argument("--output-dir", type=str, help="Path to the output directory")
    parser.add_argument(
        "--num-jobs", "-j", type=int, default=1, help="Number of parallel jobs"
    )

    args = parser.parse_args()
    prepare_genshin(args.input_dir, args.output_dir, args.num_jobs)


if __name__ == "__main__":
    main()
