import torch.nn as nn
from transformers import AutoTokenizer, ModernBertModel
import torch.nn.functional as F
import re
# from model_loading import model, tokenizer
import torch._dynamo
torch._dynamo.config.suppress_errors = True
TORCH_LOGS="+dynamo"  
TORCHDYNAMO_VERBOSE=1

class ModernBertClassifier(nn.Module):
    def __init__(self, num_labels, model_name):
        super().__init__()
        self.bert = ModernBertModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        for param in self.bert.parameters():
            param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_labels)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return self.classifier(outputs.last_hidden_state[:, 0, :])
    

def predict_sentiment(text, model, tokenizer):

    text = re.sub(r'\s+',' ',text).strip().lower()
    text = re.sub(r'[^a-z0-9\s]','',text)

    inputs = tokenizer(text, padding=True, truncation=True, max_length=1500, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        predicted_class = torch.argmax(outputs, dim=1)

    print("Predicted class:", predicted_class.item())

    emotions = ['Aggresive','Fear','Happy','Neutral','Sad']
    predicted_label = emotions[predicted_class]
    print(predicted_label)
    return predicted_label
