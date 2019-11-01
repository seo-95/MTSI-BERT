# MTSI-BERT

**MTSI-BERT** is a BERT based joint model for dialogue session classification. It was developed during my master degree thesis at [LINKS Foundation](https://linksfoundation.com/en/) under the supervision of @giusepperizzo.

##### Table of Contents  
[Why](#sec1)  
[Session](#sec2)  
[The Architecture](#sec3) 
[How to use](#sec4) 
[Dataset](#sec5)  
[Results](#sec6)  
[Dependencies](#sec7) 

<a name="sec1"/>
## Why?
MTSI-BERT goal is to extract information from the session of a multi-turn dialogue. It was developed as a joint model having three main tasks:
  - End of session detection (EOS)
  - Action classification
  - Intent classification
The action is the action that the agent has to perform on a knowledge-base in order to fulfill the user goal and it can be of two types: fetch or insert.

<a name="sec2"/>
## Session
A session is a contiguous ordered sequence of QA pairs in a multi-turn conversational scenario. MTSI-BERT takes as input a triplet of QAQ to understand the existing relation between the previous QA pair and the current Q of the user. In this way it is able to detect the end-of-session.
![](img/MTSI-input.png)

<a name="sec3"/>
# The Architecture
![](img/deep_residual.png)

<a name="sec4"/>
# How to use
## Train
To train the model:
```
python train.py
```
It will save the model dictionary into the folder:
```
savings/<TIMESTAMP>
```
and the plot of the loss into:
```
plots/
```

## Test
To test the model:
```
python test.py
```
Remember to set the path of the saved model to load in the args of the method:
```
def test(load_checkpoint_path):
```
<a name="sec5"/>
## Dataset
[KVRET](https://nlp.stanford.edu/blog/a-new-multi-turn-multi-domain-task-oriented-dialogue-dataset/)


<a name="sec6"/>
## Results
## Training losses trends
![](img/deep_losses.png)

## Test
| Attempt | #1 | #2 |
| :---: | :---: | :---: |
| Seconds | 301 | 283 |


<a name="sec7"/>
## Dependencies
- Python 3.7.4
- [Transformer package by Hugging Face](https://github.com/huggingface/transformers)
- [spaCy](https://spacy.io/)





Developed by **Matteo A. Senese** with :heart:
