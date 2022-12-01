# Emotion-Cause Relationship Between Clauses Prediction: a Novel Method Based on BERT for Emotion-Cause Pair Extraction

This repo provides the source code and data for the paper [Emotion-Cause Relationship Between Clauses Prediction: a Novel Method Based on BERT for Emotion-Cause Pair Extraction]() (2022).


<p align="center">
  <img src="example.png" width="400" title="An example of data reconstruction" alt="">
</p>

## Requirements

- Python == 3.8.5
- PyTorch == 1.7.0
- transformers == 4.5.1

##  Usage
1. Download the pertrained ["BERT-Base, Chinese"](https://github.com/google-research/bert) model
2. Construct data:

```
python ./src/utils/construct_data.py
```

3. Training the model:

```
python ./src/main.py
```

4. Evaluate results:
```
python ./src/eval.py
```


##  Acknowledgment
This repo is built upon the following work: 
```
Effective Inter-Clause Modeling for End-to-End Emotion-Cause Pair Extraction
https://github.com/Determined22/Rank-Emotion-Cause
```

Many thanks to the authors and developers!

## Others
If this work is helpful, please cite as:
```bib

```
