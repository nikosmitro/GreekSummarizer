# GreekSummarizer
An automated text summarizer for the Greek language using Deep Learning mechanisms.

The model is based on Seq2Seq encoder-decoder architecture.

#### Requirements :
Python 3.6.5 or higher.
Tensorflow 1.x

To install all required python packages run:
```bash
pip install -r requirements.txt

```
Also download spacy Greek language package :
```bash
python -m spacy download el_core_news_md

```
## Generate Summary
Input : .txt file of the text/article to be summarized.   ->  Output: .txt with its Summary.

Check the example on the example folder.

#### To generate your Summary, run :
```bash
python run_summarization.py --mode=inference --data_path=myarticle.txt --vocab_path=../vocab_v1 --log_root=logs_v1

```
## Training
#### To train the model with your data:
```bash

```
## Evaluation
#### To evaluate your model, run:
```bash

```
