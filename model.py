import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer


class BertForPostClassification(nn.Module):
    def __init__(self, model_name, num_labels, dropout, freeze_bert=False):
        super(BertForPostClassification, self).__init__()
        self.device = None
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
            nn.Linear(hidden_size, num_labels),
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
    def predict(self, x, tags):
        predictions = self.forward(x) > 0
        result = [[]] * predictions.size(0)
        index, tag_index = map(lambda x: x.tolist(), torch.where(predictions))
        for i, tag_i in zip(index, tag_index):
            result[i].append(tags[tag_i])
        return result

