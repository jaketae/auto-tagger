import torch
import transformers

model = transformers.BertForSequenceClassification.from_pretrained(
    "bert-base-uncased", num_labels=20
)

tokenizer = transformers.BertTokenizerFast.from_pretrained("bert-base-uncased")
