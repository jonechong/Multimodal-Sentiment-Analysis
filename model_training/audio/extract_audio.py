# Audio Extraction
import os
import librosa
import numpy as np
import pandas as pd
import h5py
from moviepy.editor import *


def calculate_hop_length(sr, ms):
    return int(sr * ms / 1000)


def extract_features_for_different_timesteps(y, sr):
    timesteps = [23, 100, 500, 1000]  # in milliseconds
    features_dict = {}

    for ms in timesteps:
        hop_length = calculate_hop_length(sr, ms)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, hop_length=hop_length)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length)
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=hop_length)
        tonnetz = librosa.feature.tonnetz(y=y, sr=sr, hop_length=hop_length)

        # Stack all features for the dataset
        features = np.vstack((mfcc, chroma, contrast, tonnetz))
        features_dict[f"{ms}ms"] = features.T  # Transpose to have [timesteps, features]

    return features_dict


def extract_audio_features(file_path):
    y, sr = librosa.load(file_path)
    features_dict = extract_features_for_different_timesteps(y, sr)

    # Compute the mean for a general representation (for CSV), using the default ~23ms hop_length
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
    features_mean = np.hstack(
        (
            np.mean(mfcc, axis=1),
            np.mean(chroma, axis=1),
            np.mean(contrast, axis=1),
            np.mean(tonnetz, axis=1),
        )
    )

    return features_mean, features_dict


df = pd.read_csv("../train_test_validate_split.csv")
os.makedirs("output_h5", exist_ok=True)

rows_list = []
h5_path = os.path.join("output_h5", "all_data.h5")

with h5py.File(h5_path, "w") as h5f:
    for index, row in df.iterrows():
        try:
            video_path = os.path.join(
                "../CDS datasets",
                "CMU-MOSEI",
                "Raw",
                str(row["video_id"]),
                f"{row['clip_id']}.mp4",
            )

            audio_path = f"temp_audio_{index}.wav"
            video = VideoFileClip(video_path)
            video.audio.write_audiofile(audio_path)

            # Extract features
            features_mean, features_dict = extract_audio_features(audio_path)

            # Prepare the row for the new DataFrame
            new_row = row.to_dict()
            new_row.update({f"mfcc_{i+1}": features_mean[i] for i in range(20)})
            new_row.update({f"chroma_{i+1}": features_mean[20 + i] for i in range(12)})
            new_row.update({f"contrast_{i+1}": features_mean[32 + i] for i in range(7)})
            new_row.update({f"tonnetz_{i+1}": features_mean[39 + i] for i in range(6)})
            rows_list.append(new_row)

            # Group creation in the .h5 file
            group_name = f"{row['video_id']}_{row['clip_id']}"
            if group_name in h5f:
                del h5f[group_name]
            grp = h5f.create_group(group_name)
            grp.attrs["label"] = row["annotation"]
            grp.attrs["text"] = row["text"]

            # Create datasets for different timesteps
            for timestep, features in features_dict.items():
                dataset_name = f"audio_features_{timestep}"
                grp.create_dataset(dataset_name, data=features)

            os.remove(audio_path)

        except Exception as e:
            print(f"Error processing {row['video_id']}_{row['clip_id']}: {e}")
            if group_name in h5f:
                del h5f[group_name]
            continue


columns_order = [col for col in new_row.keys() if col not in ["annotation"]] + [
    "annotation"
]
new_df = pd.DataFrame(rows_list, columns=columns_order)

# Save to new CSV
new_df.to_csv("audio_features.csv", index=False)
## Normalize
h5_path = "output_h5/all_data.h5"

timesteps = ["23ms", "100ms", "500ms", "1000ms"]
global_stats = {timestep: {"mean": None, "std": None} for timestep in timesteps}

all_features = {timestep: [] for timestep in timesteps}

with h5py.File(h5_path, "r") as h5f:
    for group_name in h5f:
        group = h5f[group_name]
        for timestep in timesteps:
            dataset_name = f"audio_features_{timestep}"
            features = group[dataset_name][:]
            all_features[timestep].append(features)

# Compute global statistics for each timestep
for timestep in timesteps:
    features_concat = np.concatenate(all_features[timestep], axis=0)
    global_stats[timestep]["mean"] = np.mean(features_concat, axis=0)
    global_stats[timestep]["std"] = np.std(features_concat, axis=0)
new_h5_path = "output_h5/all_data_normalized.h5"


def apply_global_normalization(features, mean, std):
    # Avoid division by zero
    std_replaced = np.where(std == 0, 1, std)
    return (features - mean) / std_replaced


with h5py.File(h5_path, "r") as h5f_original:
    with h5py.File(new_h5_path, "w") as h5f_normalized:
        for group_name in h5f_original:
            group_original = h5f_original[group_name]
            group_normalized = h5f_normalized.create_group(group_name)

            for timestep in timesteps:
                dataset_name = f"audio_features_{timestep}"
                original_features = group_original[dataset_name][:]
                mean = global_stats[timestep]["mean"]
                std = global_stats[timestep]["std"]
                normalized_features = apply_global_normalization(
                    original_features, mean, std
                )

                normalized_dataset_name = f"{dataset_name}_normalized"
                group_normalized.create_dataset(
                    normalized_dataset_name, data=normalized_features
                )
