import torch.nn as nn
from transformers import AutoTokenizer, ModernBertModel
import torch.nn.functional as F

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
