## Attention Realignment and Pseudo-Labelling for InterpretableCross-Lingual Classification of Crisis Tweets

**Purposel**: A custom cross-lingual neural network model over XLM-R with the capability to attend over the same words (```dlo``` in Haitian Creolel versus ```water``` in English) in different languages.

### Paper/Cite
http://kiml2020.aiisc.ai/papers/KiML2020_paper_8.pdf (To appear at [KiML@KDD'20](http://kiml2020.aiisc.ai/index.html))

### Requirements
- Python3.6, Keras, Tensorflow.
- Install [fairseq](https://github.com/pytorch/fairseq) for XLMR. Apex is not needed.

### Data
Download Appen [dataset](https://appen.com/datasets/combined-disaster-response-data/) consisting of Multilingual Disaster Response Messages.

### Extract XLM-R embeddings
- ```python get_xlmr_embeddings.py en train```
- ```python get_xlmr_embeddings.py en val```
- ```python get_xlmr_embeddings.py en test```
- ```python get_xlmr_embeddings.py ml train ```
- ```python get_xlmr_embeddings.py en val```
- ```python get_xlmr_embeddings.py en test```
This step produces 6 ```.npy``` files with embeddings and 6 ```.txt``` files with corresponding tweets. This will make it easier to train as XLMR is a bit slow.

### Running Models (en --> ml)
##### Baseline
```python baseline.py en ml```
##### Model A
```python modelA.py en ml```
##### Model B
```python modelB.py en ml```

### Results (Accuracy in %)
Source --> Target (Source --> Source)

| S --> T | Baseline | Model A | Model B|
 :-: |  :-: |  :-: |  :-: 
| en --> ml | 59.98 (80.57) | 62.53 (77.02) | 66.79 (82.39) |
| ml --> en | 60.93 (70.07) | 65.69 (63.50) | 70.95 (73.84) |

### Attention Visualization
[Click Here](https://github.com/jitinkrishnan/Cross-Lingual-Crisis-Tweet-Classification/blob/master/Attention%20Plot%20Example.ipynb) to view the Jupyter Notebook that shows the attention heat map.

### Contact information
For help or issues, please submit a GitHub issue or contact Jitin Krishnan (`jkrishn2@gmu.edu`).
