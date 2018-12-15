# Neural Machine Translation

DS-GA-1011 Final Project

### Project Description

In this project, we built a machine translation system using deep learning techniques. There are two language pairs: Viet- namese to English and Chinese to English. We used sequence-to-sequence framework as our prototype, and implement Recurrent Neural Network as well as Convolutional Neural Network as encoder/decoder. We also incorporated some modern deep learning techniques including Attention Mechanism and Teacher Forcing. The translation result was evalu- ated using the Bilingual Evaluation Understudy Score, or BLEU for short, which is a commonly-used metric to compare generated sentences and reference sentences.


### Current Results (On Validation Set)

| Language  | Encoder | Decoder | BLEU |
| ------------- | ------------- |------------- | ------------- |
| Ch-En  | RNN | RNN no Attn | 6.86 |
| Vi-En  | RNN | RNN no Attn | 7.58 |
| Ch-En  | RNN | RNN Attn | 13.02 |
| Ch-En  | RNN | RNN Attn | 12.46 (with beam search k=3)|
| Vi-En  | RNN | RNN Attn | 15.87 |
| Ch-En  | CNN | RNN Attn | 7.91 |
| Vi-En  | CNN  | RNN Attn | 8.51 |

### Current Results (On Test Set)
Use the best model based on validation set BLEU score

| Language  | Encoder | Decoder | BLEU |
| ------------- | ------------- |------------- | ------------- |
| Ch-En  | RNN | RNN Attn | 12.21 |
| Vi-En  | RNN | RNN Attn | 15.47 |

### To-do-list

Although we have already achieved our objectives for the final project. We believe there are still rooms for future improvement in this project, and we'd like to continue working on it even after the course ends. The potential future work includes:  

- [ ] Change GRU to LSTM
- [ ] Try deeper netowrks for CNN and RNN
- [ ] Incorporate pre-trained embeddings

### References

Our code referred to: 

https://github.com/spro/practical-pytorch

https://github.com/AuCson/PyTorch-Batch-Attention-Seq2seq

https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html

DS-GA-1011 Fall 2018, Lab 8



