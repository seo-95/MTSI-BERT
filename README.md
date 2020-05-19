# MTSI-BERT

**MTSI-BERT** is a BERT based joint model for dialogue session classification. It was developed during my master degree thesis at [LINKS Foundation](https://linksfoundation.com/en/) under the supervision of @giusepperizzo.

##### Table of Contents  
[Why](#sec1)  
[Session](#sec2)  
[The Architecture](#sec3)  
[How to use](#sec4)  
[Hyperparamters](#sec5)   
[Dataset](#sec6)  
[Results](#sec7)  
[Dependencies](#sec8) 

<a name="sec1"/>

## Why?
MTSI-BERT goal is to extract information from the session of a multi-turn dialogue. It was developed as a joint model having three main tasks:
  - *End of session detection* (EOS)
  - *Action classification* for the session: corresponds to insert/fetch operations on a knowledge-base to fullfill the user goal for the session
  - *Intent classification* for the session


<a name="sec2"/>

## Session
A session is a contiguous ordered sequence of QA pairs in a multi-turn conversation. MTSI-BERT takes as input a triplet of QAQ to understand the existing relation between the previous QA pair and the current Q of the user. In this way it is able to detect the end-of-session.<br>


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

## Hyperparameters

| **Parameter** | **Value** |
| :---: | :---: |
| **Mini-batch** | 16 |
| **BERT lr** | 5e-5 |
| **NN lr** | 1e-3 |
| **Weight decay** | 0.1 |
| **Milestones** | 5, 10, 15, 20, 30, 40, 50, 75 |
| **Gamma** | 0.5 |


<a name="sec6"/>

## Dataset

[KVRET](https://nlp.stanford.edu/blog/a-new-multi-turn-multi-domain-task-oriented-dialogue-dataset/)


<a name="sec7"/>

## Results

## Training losses trends
![](img/deep_losses.png)


## Test

### End of session
| **Model** | **Precision** | **Recall** | **F1** |
| :---: | :---: | :---: | :---: |
| **MTSI-BERT** | 0.9915 ± 0.0003 | 0.9962 ± 0.0008 | 0.9938 ± 0.0005 |
| **Reference** | 0.9558 ± 0.0016 | 0.9659 ± 0.0003 | 0.9638 ± 0.0006 |

### Action
| **Model** | **Precision** | **Recall** | **F1** |
| :---: | :---: | :---: | :---: |
| **MTSI-BERT** | 1.00 | 1.00 | 1.00 |
| **Reference** | 0.9980 | 0.9895 | 0.9937 |

### Intent
| **Model** | **Precision** | **Recall** | **F1** |
| :---: | :---: | :---: | :---: |
| **MTSI-BERT** | 1.00 | 1.00 | 1.00 |
| **Reference** | 1.00 | 1.00 | 1.00 |


<a name="sec8"/>  

## Dependencies
- Python 3.7.4
- [Transformer package by Hugging Face](https://github.com/huggingface/transformers)
- [spaCy](https://spacy.io/)


## References
paper: [url](http://www.lrec-conf.org/proceedings/lrec2020/pdf/2020.lrec-1.89.pdf)<br>
If you use this work please cite
```
@InProceedings{senese-EtAl:2020:LREC,
  author    = {Senese, Matteo Antonio  and  Rizzo, Giuseppe  and  Dragoni, Mauro  and  Morisio, Maurizio},
  title     = {MTSI-BERT: A Session-aware Knowledge-based Conversational Agent},
  booktitle      = {Proceedings of The 12th Language Resources and Evaluation Conference},
  month          = {May},
  year           = {2020},
  address        = {Marseille, France},
  publisher      = {European Language Resources Association},
  pages     = {710--718},
  abstract  = {In the last years, the state of the art of NLP research has made a huge step forward. Since the release of ELMo (Peters et al., 2018), a new race for the leading scoreboards of all the main linguistic tasks has begun. Several models have been published achieving promising results in all the major NLP applications, from question answering to text classification, passing through named entity recognition. These great research discoveries coincide with an increasing trend for voice-based technologies in the customer care market. One of the next biggest challenges in this scenario will be the handling of multi-turn conversations, a type of conversations that differs from single-turn by the presence of multiple related interactions. The proposed work is an attempt to exploit one of these new milestones to handle multi-turn conversations. MTSI-BERT is a BERT-based model achieving promising results in intent classification, knowledge base action prediction and end of dialogue session detection, to determine the right moment to fulfill the user request. The study about the realization of PuffBot, an intelligent chatbot to support and monitor people suffering from asthma, shows how this type of technique could be an important piece in the development of future chatbots.},
  url       = {https://www.aclweb.org/anthology/2020.lrec-1.89}
}

```
