import torch.nn.functional as F
from torch import nn
from transformers import AutoModel


class BertForPostClassification(nn.Module):
    def __init__(self, model_name, num_labels, dropout, freeze_bert=False):
        super(BertForPostClassification, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.bert.config.return_dict = False
        self.bert.config.add_cross_attention = False
        self.bert.config.output_attentions = False
        self.bert.config.output_hidden_states = False
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        hidden_size = self.bert.config.hidden_size
        self.classier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(hidden_size, num_labels),
        )

    def forward(self, **tokens):
        bert_out = self.bert(**tokens)
        clf_tokens = bert_out[0][:, 0, :]
        logits = self.classier(clf_tokens)
        return logits

