import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer

from utils import chunkify


class BertForPostClassification(nn.Module):
    def __init__(
        self, model_name, tags, max_len, min_len, dropout=0.2, freeze_bert=False
    ):
        super(BertForPostClassification, self).__init__()
        self.device = None
        self.tags = tags
        self.max_len = max_len
        self.min_len = min_len
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(
            model_name,
            return_dict=False,
            add_cross_attention=False,
            output_attentions=False,
            output_hidden_states=False,
        )
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        hidden_size = self.bert.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(hidden_size, len(tags)),
        )

    def to(self, device):
        self.device = device
        return super(BertForPostClassification, self).to(device)

    def forward(self, x):
        assert self.device, "Have you called `.to(device)` on this model?"
        tokens = self.tokenizer(
            list(x), truncation=True, padding=True, return_tensors="pt"
        ).to(self.device)
        bert_out = self.bert(**tokens)
        clf_tokens = bert_out[0][:, 0, :]
        logits = self.classifier(clf_tokens)
        return logits

    @torch.no_grad()
    def predict(self, x):
        self.eval()
        chunks = chunkify(x)
        logits = self.forward(chunks)
        mean_logits = logits.mean(dim=0)
        index = torch.where(mean_logits > 0)[0]
        result = [self.tags[i] for i in index]
        return result

    @property
    def config(self):
        return {
            "min_len": self.min_len,
            "max_len": self.max_len,
        }
