import logging
import shutil
import subprocess
import time
import warnings
from pathlib import Path

import gtts
import librosa
import numpy as np
import pandas as pd
import shortuuid
import soundfile
import srsly
from pydub import AudioSegment
import argparse

logging.getLogger().setLevel(logging.INFO)
NUM_SELF_CHATS = 5


def create_self_chat_episodes(num_self_chats, output_file):
    subprocess.call(
        [
            "parlai",
            "self_chat",
            "--model-file",
            "zoo:blender/blender_90M/model",
            "--task",
            "convai2",
            "--inference",
            "topk",
            "--num-self-chats",
            str(num_self_chats),
            "--display-examples",
            "True",
            "--datatype",
            "valid",
            "--outfile",
            output_file,
        ]
    )


def format_episode_transcript(episode_raw):
    # as a dataframe
    episode = []
    episode_id = shortuuid.uuid()
    for e in episode_raw["dialog"]:
        episode.extend([{"id": ee["id"], "text": ee["text"]} for ee in e])

    return (
        pd.DataFrame(episode)
        .reset_index()
        .rename(mapper={"index": "exchange_index"}, axis="columns")
        .assign(speaker=lambda x: x.id.apply(lambda y: int(y.split("_")[1])))
        .assign(episode_id=episode_id)
    )


def synthesize_tts_episode(formatted_episode_df, output_dir):
    # given an episode DF, synthesize audio for each utterance
    for idx, e in formatted_episode_df.iterrows():
        time.sleep(1)  # prevent IP ban?
        save_path = output_dir / f"{e.exchange_index}_speaker_{e.speaker}.mp3"

        # alternative voices, useful for debugging > could be improved with more variance
        if e.speaker == 1:
            tts = gtts.gTTS(e.text, lang="en", tld="com", slow=True)
        elif e.speaker == 2:
            tts = gtts.gTTS(e.text, lang="en", tld="ca", slow=True)

        tts.save(save_path)
    logging.info(f"Saved TTS audio fragments to {str(output_dir)}")


def retrieve_episode_audio(output_dir):
    audio_fragment_records = []
    for file in list(output_dir.glob("./*.mp3")):
        with warnings.catch_warnings():
            # pysoundfile warnings
            warnings.filterwarnings("ignore")
            y, s = librosa.load(str(file))  # FYI: assigns default sample rate
            audio_fragment_records.append(
                {"file": file.name, "sample_array": y, "sample_array_shape": y.shape[0]}
            )

    audio_fragment_df = (
        pd.DataFrame(audio_fragment_records)
        # probably just the df index; but to be sure
        .assign(sequence_idx=lambda x: x.file.apply(lambda y: int(y[0])))
        .sort_values("sequence_idx")
        # speaker as channel
        .assign(
            channel=lambda x: x.sequence_idx.apply(lambda y: 1 if y % 2 == 0 else 2)
        )
        .reset_index(drop=True)
    )
    logging.info(
        f"Retrieved {audio_fragment_df.shape[0]} audio fragments from {str(output_dir)}"
    )
    return audio_fragment_df


def collate_channel_audio(audio_fragment_df, output_dir):
    channel_1_segments = []
    channel_2_segments = []
    for idx, e in audio_fragment_df.iterrows():

        if e.channel == 1:
            channel_1_segments.append(e.sample_array)
            # pad alternating channel (channel 2) with equivalent size zero array to create interleave
            channel_2_segments.append(
                np.zeros(e.sample_array.shape[0], dtype=np.float32)
            )
        else:
            channel_2_segments.append(e.sample_array)
            channel_1_segments.append(
                np.zeros(e.sample_array.shape[0], dtype=np.float32)
            )

    default_sr = 22050
    channel_1_padded = np.concatenate(channel_1_segments)
    soundfile.write(output_dir / "channel_1_temp.wav", channel_1_padded, default_sr)

    channel_2_padded = np.concatenate(channel_2_segments)
    soundfile.write(output_dir / "channel_2_temp.wav", channel_2_padded, default_sr)
    logging.info(
        f"Left/right channel WAV audio collated and saved to {str(output_dir)}"
    )


def sum_left_right_mono_audio(output_dir):
    # consolidate into an interleaving, channel seperated source
    left_channel = AudioSegment.from_wav(output_dir / "channel_1_temp.wav")
    right_channel = AudioSegment.from_wav(output_dir / "channel_2_temp.wav")

    stereo_sound = AudioSegment.from_mono_audiosegments(left_channel, right_channel)
    stereo_sound.export(output_dir / "consolidated_final.wav")
    logging.info(f"Left/right channel WAV audio summed and saved to {str(output_dir)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="synthesize_transcript_audio",
        description="Synthesize stereo audio in the style of a call-centre exchange",
    )
    parser.add_argument("output_dir", type=str, help="Output dir")
    args = parser.parse_args()

    # create/null output dir
    base_output_dir = Path(args.output_dir)
    if not base_output_dir.exists():
        raise ValueError("output_dir does not exist")
    shutil.rmtree((str(base_output_dir))) if base_output_dir.exists() else None
    base_output_dir.mkdir(parents=True, exist_ok=True)
    (base_output_dir / "final_calls").mkdir()

    # create self-chat episodes
    self_chat_output = str(base_output_dir / "self_chat_episodes.jsonl")
    create_self_chat_episodes(NUM_SELF_CHATS, self_chat_output)

    all_episodes_raw = list(srsly.read_jsonl(self_chat_output))
    logging.info(f"{len(all_episodes_raw)} self-chat episodes created")
    for episode in all_episodes_raw:
        # format transcript
        formatted_episode = format_episode_transcript(episode)
        episode_output_dir = base_output_dir / formatted_episode.iloc[0].episode_id
        episode_output_dir.mkdir(
            exist_ok=True, parents=True
        ) if episode_output_dir.exists() == False else None
        formatted_episode.to_csv(
            episode_output_dir / "episode_transcript.csv", index=False
        )

        # synthesize audio
        synthesize_tts_episode(formatted_episode, episode_output_dir)

        # splice audio
        audio_fragment_df = retrieve_episode_audio(episode_output_dir)
        collate_channel_audio(audio_fragment_df, episode_output_dir)
        sum_left_right_mono_audio(episode_output_dir)

        # move consolidated
        shutil.move(
            str(episode_output_dir / "consolidated_final.wav"),
            base_output_dir / f"final_calls/{episode_output_dir.name}.wav",
        )
