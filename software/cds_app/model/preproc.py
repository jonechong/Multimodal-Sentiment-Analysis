import os
import cv2
import numpy as np
import pandas as pd
import h5py
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model

import librosa
from moviepy.editor import *
import speech_recognition as sr

import torch
from transformers import BertTokenizer, BertModel


def proc_audio(path):
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
    # Extract audio and save temporarily
    # audio_path = f"../temp"
    # video = VideoFileClip(path)
    # video.audio.write_audiofile(audio_path,codec="pcm_s16le")
    audio_path = f"temp_audio.wav"
    video = VideoFileClip(path)
    video.audio.write_audiofile(audio_path,codec="pcm_s16le")

    # Extract features
    features_mean, features_dict = extract_audio_features(audio_path)
    return features_mean, features_dict

def proc_face(path):
    # Load the pre-trained ResNet50 model
    base_model = ResNet50(weights='imagenet', include_top=False)
    model = Model(inputs=base_model.input, outputs=base_model.output)
    # Modify pooling layer to global average pooling to get fixed size output. This results in a output vector of size 2048
    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    model = Model(inputs=base_model.input, outputs=x)

    # Load OpenCV's Haar Cascade for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def extract_facial_features(frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        if len(faces) == 0:
            return None  # Return None if no faces are detected

        (x, y, w, h) = faces[0]  # Consider the first face
        face = frame[y:y+h, x:x+w]
        face = cv2.resize(face, (224, 224))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = np.expand_dims(face, axis=0)
        face = preprocess_input(face)
        features = model.predict(face)
        return features.flatten()
    
    frame_skip = 30
    cap = cv2.VideoCapture(path)
    features_per_frame = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if cap.get(cv2.CAP_PROP_POS_FRAMES) % frame_skip == 0:
            features = extract_facial_features(frame)
            if features is not None:
                features_per_frame.append(features)
    cap.release()
    if features_per_frame:
        # Calculate the average of the features
        average_features = np.mean(features_per_frame, axis=0)
        return average_features

# def proc_text(path):
#     # Initialize the BERT tokenizer and model
#     tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#     model = BertModel.from_pretrained('bert-base-uncased')

#     # Function to encode text to BERT features with variable max_length
#     def encode_text_for_bert(text, max_length):
#         inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
#         with torch.no_grad():
#             outputs = model(**inputs)
#         embeddings = outputs.last_hidden_state
#         feature_vector = torch.mean(embeddings, dim=1)
#         return feature_vector.squeeze().cpu().numpy()
#     bert_features = encode_text_for_bert(row['text'], 512)
#     return bert_features

def proc_text(path):
    def extract_text():
        # Initialize recognizer class                                       
        r = sr.Recognizer()
        # audio object                                                         
        audio = sr.AudioFile("temp_audio.wav")
        #read audio object and transcribe
        with audio as source:
            audio = r.record(source)                  
            result = r.recognize_google(audio)
        return result
    # Initialize the BERT tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    # Function to encode text to BERT features with variable max_length
    def encode_text_for_bert(text, max_length):
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings = outputs.last_hidden_state
        feature_vector = torch.mean(embeddings, dim=1)
        return feature_vector.squeeze().cpu().numpy()
    bert_features = encode_text_for_bert(extract_text(), 512)
    return bert_features