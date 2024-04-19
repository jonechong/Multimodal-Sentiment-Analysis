import numpy as np
import librosa
import os
import pandas as pd
import h5py
from moviepy.editor import VideoFileClip

df = pd.read_csv("../train_test_validate_split.csv")


def extract_mel_spectrogram(file_path, sr=22050, n_mels=128):
    y, _ = librosa.load(file_path, sr=sr)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    S_dB = librosa.power_to_db(S, ref=np.max)
    return S_dB


total_sum = 0
total_sq_sum = 0
total_count = 0

for index, row in df.iterrows():
    audio_path = f"temp_audio_{index}.wav"
    try:
        video_path = os.path.join(
            "../CDS datasets",
            "CMU-MOSEI",
            "Raw",
            str(row["video_id"]),
            f"{row['clip_id']}.mp4",
        )
        video = VideoFileClip(video_path)
        video.audio.write_audiofile(audio_path)

        mel_spectrogram = extract_mel_spectrogram(audio_path)

        total_sum += mel_spectrogram.sum()
        total_sq_sum += (mel_spectrogram**2).sum()
        total_count += mel_spectrogram.size

    except Exception as e:
        print(f"Failed to process {row['video_id']}_{row['clip_id']} due to error: {e}")
    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)

global_mean = total_sum / total_count
global_std = np.sqrt((total_sq_sum / total_count) - (global_mean**2))

h5_output_path = r"output_h5/mels.h5"
os.makedirs(os.path.dirname(h5_output_path), exist_ok=True)

with h5py.File(h5_output_path, "w") as h5f:
    for index, row in df.iterrows():
        audio_path = f"temp_audio_{index}.wav"
        try:
            video_path = os.path.join(
                "../CDS datasets",
                "CMU-MOSEI",
                "Raw",
                str(row["video_id"]),
                f"{row['clip_id']}.mp4",
            )
            video = VideoFileClip(video_path)
            video.audio.write_audiofile(audio_path)

            mel_spectrogram = extract_mel_spectrogram(audio_path)
            normalized_mel_spectrogram = (mel_spectrogram - global_mean) / global_std

            group_name = f"{row['video_id']}_{row['clip_id']}"
            grp = h5f.create_group(group_name)
            grp.attrs["label"] = row["annotation"]
            grp.attrs["text"] = row["text"]

            grp.create_dataset("mels", data=mel_spectrogram)
            grp.create_dataset("normalized_mels", data=normalized_mel_spectrogram)

        except Exception as e:
            print(
                f"Failed to process {row['video_id']}_{row['clip_id']} due to error: {e}"
            )
        finally:
            if os.path.exists(audio_path):
                os.remove(audio_path)
