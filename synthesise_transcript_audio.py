from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile
import logging

logging.getLogger().setLevel(logging.INFO)



def retrieve_episode_audio(output_dir):
    audio_fragment_records = []
    for file in list(output_dir.glob("./*.mp3")):
        # collate utterance audio files into raw samples
        y, s = librosa.load(str(file))  # FYI: assigns default sample rate
        audio_fragment_records.append(
            {"file": file.name, "sample_array": y, "sample_array_shape": y.shape[0]}
        )

    audio_fragment_df = (
        pd.DataFrame(audio_fragment_records)
        # probably just the df index; but to be sure
        .assign(sequence_idx=lambda x: x.file.apply(lambda y: int(y.split("_")[-1][0])))
        .sort_values("sequence_idx")
        # speaker as channel
        .assign(
            channel=lambda x: x.sequence_idx.apply(lambda y: 1 if y % 2 == 0 else 2)
        )
        .reset_index(drop=True)
    )
    logging.info(f"Retrieved {audio_fragment_df.shape[0]} audio fragments from {output_dir.name}")
    return audio_fragment_df


def collate_channel_audio(audio_fragment_df, output_dir):
    # pad channel 1/2 chunks to ensure for interleaving pattern
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
            # odd indices are channel 2
            channel_2_segments.append(e.sample_array)
            # otherwise, channel 2 length zero array
            channel_1_segments.append(
                np.zeros(e.sample_array.shape[0], dtype=np.float32)
            )

    # temp save for channel 1/2 audio - saves as mono
    default_sr = 22050
    channel_1_padded = np.concatenate(channel_1_segments)
    soundfile.write(output_dir / "channel_1_temp.wav", channel_1_padded, default_sr)

    channel_2_padded = np.concatenate(channel_2_segments)
    soundfile.write(output_dir / "channel_2_temp.wav", channel_2_padded, default_sr)
    logging.info(f"Left/right channel WAV audio collated and saved to {output_dir.name}")

def sum_left_right_mono_audio(output_dir):
    # consolidate into an interleaving, channel seperated source
    left_channel = AudioSegment.from_wav(output_dir / "channel_1_temp.wav")
    right_channel = AudioSegment.from_wav(output_dir / "channel_2_temp.wav")

    stereo_sound = AudioSegment.from_mono_audiosegments(left_channel, right_channel)
    stereo_sound.export(output_dir / "consolidated_final.wav")
    logging.info(f"Left/right channel WAV audio summed and saved to {output_dir.name}")


if __name__ == "__main__":
    output_dir = Path("./output/synth_calls/sample_transcript")
    audio_fragment_df = retrieve_episode_audio(output_dir)
    collate_channel_audio(audio_fragment_df, output_dir)
    sum_left_right_mono_audio(output_dir)