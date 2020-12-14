import torch.nn.functional as F
from torch import nn
from transformers import AutoModel, AutoTokenizer


class BertForPostClassification(nn.Module):
    def __init__(self, model_name, num_labels, dropout, freeze_bert=False):
        super(BertForPostClassification, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(
            model_name,
            return_dict=False,
            add_cros_attention=False,
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

    def forward(self, x):
        tokens = self.tokenizer(x)
        bert_out = self.bert(**tokens)
        clf_tokens = bert_out[0][:, 0, :]
        logits = self.classifier(clf_tokens)
        return logits

