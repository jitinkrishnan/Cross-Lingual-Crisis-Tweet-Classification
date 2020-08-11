## Attention Realignment and Pseudo-Labelling for InterpretableCross-Lingual Classification of Crisis Tweets

**Purposel**: A custom cross-lingual neural network model over XLM-R with the capability to attend over the same words (```dlo``` in Haitian Creolel versus ```water``` in English) in different languages.

### Requirements
- Python3.6, Keras, Tensorflow.
- Install [fairseq](https://github.com/pytorch/fairseq) for XLMR. Apex is not needed.

### Data
Download Appen [dataset](https://appen.com/datasets/combined-disaster-response-data/) consisting of Multilingual Disaster Response Messages.

### Extract XLM-R embeddings
This step caches the embeddings and produces 6 .npy files and 6 text files with corresponding sentences.
```python get_xlmr_embeddings.py en train```
```python get_xlmr_embeddings.py en val```
```python get_xlmr_embeddings.py en test```
```python get_xlmr_embeddings.py ml train ```
```python get_xlmr_embeddings.py en val```
```python get_xlmr_embeddings.py en test```

### Running Models (en --> ml)
##### Baseline
```python baseline.py en ml```
##### Model A
```python modelA.py en ml```
##### Model B
```python modelB.py en ml```

### Attention Visualization
[Click Here](https://github.com/jitinkrishnan/Cross-Lingual-Crisis-Tweet-Classification/blob/master/Attention%20Plot%20Example.ipynb) to view the Jupyter Notebook that shows the attention heat map.

### Contact information
For help or issues, please submit a GitHub issue or contact Jitin Krishnan (`jkrishn2@gmu.edu`).
