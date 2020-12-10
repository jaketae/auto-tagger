import torch
from torch import nn
import transformers as hf


class DistilBertPostTagger(nn.Module):
    def __init__(self, hidden_dim, num_tags, dropout=0.5):
        super(DistilBertPostTagger, self).__init__()
        self.bert = hf.DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.preclassifier = nn.Linear(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_tags)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        bert_out = self.bert(x)
        hidden_state = distilbert_output[0]
        pooled_output = hidden_state[:, 0]
