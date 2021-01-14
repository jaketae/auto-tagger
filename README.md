# Auto Tagger

Auto Tagger is a BERT fine-tuner that can automatically generate tags for posts on my [study blog](http://jaketae.github.io/).

## Motivation

While maintaining my study blog, I realized that tag attribution was a multi-label classification task that could potentially be automated. For instance, the standard YAML format for tagging in Jekyll looks as follows:

```
tags:
  - deep_learning
  - pytorch
```

 Since the blog was already maintained through a semi-automated publication pipeline in which posts were converted from Jupyter notebooks to markdown, this project was conceived as a useful addition to that preexisting workflow, with the goal of automatic blog tag attribution through the use of BERT and other variant transformer models.

## Requirements

The project can be subdivided into two segments. The first segment concerns data collection and preprocessing. This process requires the following dependencies.

```
beautifulsoup4==4.9.1
pandas==1.1.3
requests==2.24.0
scikit-learn==0.23.2
tqdm==4.49.0
```

The model experimentation and training portion of the project requires the following:

```
pytorch==1.6.0
transformers==3.5.1
```

All dependencies are specified in `requirements.txt`.

## Directory

Raw labeled datasets scraped from the website reside in the `./data/` directory. The script also expects a `./checkpoints/` directory to be able to save and load model weights. Below is a tree directory that demonstrates a sample structure.

```
.
├── checkpoints
│   ├── roberta-unfreeze.json
│   └── roberta-unfreeze.pt
├── data
│   ├── test.csv
│   ├── train.csv
│   └── val.csv
├── dataset.py
├── eda.ipynb
├── logs
├── model.py
├── requirements.txt
├── scrape.py
├── test.py
├── train.py
└── utils.py
```

## Usage

The repository comes with convenience scripts to allow for training, saving, and testing different transformer models.

### Training

The example below demonstrates how to train a RoBERTa model with minimal custom configurations.

```
python train.py --model_name="roberta-base" --save_title="roberta-unfreeze" --unfreeze_bert --num_epochs=20 --batch_size=32
```

The full list of training arguments are provided below:

```
usage: train.py [-h]
                [--model_name {bert-base,distilbert-base,roberta-base,distilroberta-base,allenai/longformer-base-4096}]
                [--save_title SAVE_TITLE] [--load_title LOAD_TITLE]
                [--num_epochs NUM_EPOCHS] [--log_interval LOG_INTERVAL]
                [--batch_size BATCH_SIZE] [--patience PATIENCE]
                [--max_len MAX_LEN] [--min_len MIN_LEN] [--freeze_bert]
                [--unfreeze_bert]

optional arguments:
  -h, --help            show this help message and exit
  --model_name {bert-base,distilbert-base,roberta-base,distilroberta-base,allenai/longformer-base-4096}
  --save_title SAVE_TITLE
  --load_title LOAD_TITLE
  --num_epochs NUM_EPOCHS
  --log_interval LOG_INTERVAL
  --batch_size BATCH_SIZE
  --patience PATIENCE
  --max_len MAX_LEN     maximum length of each text
  --min_len MIN_LEN     minimum length of each text
  --freeze_bert
  --unfreeze_bert
```

### Testing

The example below demonstrates how to test a RoBERTa model whose weights were saved as ``"roberta-unfreeze"``.

```
python test.py --model_name="roberta-base" --save_title="roberta-unfreeze" --batch_size=32 
```

The full list of testing arguments are provided below:

```
usage: test.py [-h]
               [--model_name {bert-base,distilbert-base,roberta-base,distilroberta-base,allenai/longformer-base-4096}]
               [--max_len MAX_LEN] [--min_len MIN_LEN]
               [--save_title SAVE_TITLE] [--batch_size BATCH_SIZE]

optional arguments:
  -h, --help            show this help message and exit
  --model_name {bert-base,distilbert-base,roberta-base,distilroberta-base,allenai/longformer-base-4096}
  --max_len MAX_LEN     maximum length of each text
  --min_len MIN_LEN     minimum length of each text
  --save_title SAVE_TITLE
                        title of saved file
  --batch_size BATCH_SIZE
```

## License

Released under the [MIT License](https://github.com/jaketae/auto-tagger/blob/master/LICENSE).