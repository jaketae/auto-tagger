import torch.nn.functional as F
from torch import nn
from transformers import AutoModel


class BertForPostClassification(nn.Module):
    def __init__(self, model_name, num_labels, dropout, freeze_bert):
        super(BertForPostClassification, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        hidden_size = self.bert.config.hidden_size
        self.pre_classifier = nn.Linear(hidden_size, hidden_size)
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, **tokens):
        bert_out = self.bert(**tokens)
        try:
            pooler_output = bert_out["pooler_output"]
        except KeyError:
            pooler_output = bert_out["last_hidden_state"][:, 0]
        pooled_output = F.relu(self.dropout(self.pre_classifier(pooler_output)))
        logits = self.classifier(pooled_output)
        return logits

