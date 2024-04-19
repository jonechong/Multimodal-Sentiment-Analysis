import torch
from torchvision import transforms
from PIL import Image
from .preproc import proc_audio, proc_face, proc_text
from .multimodel import MultimodalSentimentModel
import os

label_mapping = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}

def preprocess_video(path):
    audioproc_mean, audioproc_dict = proc_audio(path)
    faceproc = proc_face(path)
    textproc = proc_text(path)
    return audioproc_mean, faceproc, textproc, audioproc_dict


def predict_sentiment(video_file):
    # Hyperparams
    batch_size = 64
    num_epochs = 20
    audio_feature_dim = 45 
    facial_feature_dim = 2048 
    text_feature_dim = 768 
    encoding_dim = 512

    learning_rate = 1e-5
    dropout_rate = 0.1

    # Other definitions
    h5_file = r'combined_features.h5'
    output_dim = 3  
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "final.pth")

    # Preprocess the video file
    audioproc_mean, faceproc, textproc, audioproc_dict = preprocess_video(video_file)
    
    # Convert video data to PyTorch tensor
    audio_features = torch.tensor(audioproc_mean, dtype=torch.float32)
    facial_features = torch.tensor(faceproc, dtype=torch.float32)
    text_features = torch.tensor(textproc, dtype=torch.float32) # Modify as per your data format
    
    # If GPU available, move tensor to GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # video_tensor = video_tensor.to(device)
    audio_features = audio_features.to(device)
    facial_features = facial_features.to(device)
    text_features = text_features.to(device)

    # Load the model
    model = MultimodalSentimentModel(audio_feature_dim=audio_feature_dim,
    facial_feature_dim=facial_feature_dim,
    text_feature_dim=text_feature_dim,
    encoding_dim=encoding_dim,
    output_dim=output_dim,
    dropout_rate=dropout_rate)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')), strict=False)
    model.to(device)
    model.eval()  # Set model to evaluation mode
    
    # Perform inference
    with torch.no_grad():
        # Forward pass
        outputs = model(audio_features, facial_features, text_features)
        
        # Post-process outputs if necessary
        # For example, apply softmax if the model outputs logits
        
        # Get predicted sentiment
        predicted_sentiment = outputs.argmax().item()  # Modify based on your output format
        predicted_sentiment_label = label_mapping[predicted_sentiment]

    return predicted_sentiment_label
