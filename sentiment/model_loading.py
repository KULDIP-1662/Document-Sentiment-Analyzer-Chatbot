import torch
import dill
import torch.nn.functional as F
from transformers import AutoTokenizer

model = torch.load(r'C:\Users\Kulde\OneDrive\Desktop\OFFICE\Emotion_Sentiment_Analysis\models\CLEANED\model_dill_2_layer_Cleandata.pth', pickle_module=dill,map_location=torch.device('cpu'))
tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")