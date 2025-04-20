import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# MAX_LENGTH = 1500
MAX_LENGTH = 100    #delete
BATCH_SIZE = 10 #delete
# BATCH_SIZE = 64
EPOCHS = 1  #delete
# EPOCHS = 20
MODEL_NAME = "answerdotai/ModernBERT-base"
CHECKPOINT_DIR = "checkpoints"
DATA_PATH = "C:/Users/Kulde/OneDrive/Desktop/OFFICE/Emotion_Sentiment_Analysis/test_data.csv"
FINE_TUNED_MODEL_PATH = "C:/Users/Kulde/OneDrive/Desktop/OFFICE/Emotion_Sentiment_Analysis/models/model.pth"