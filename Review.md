# LyS at SemEval 2024: Prototype for the Task 3


## About our model

Our system consits of an end-to-end neural architecture onformed by a pretrained encoder with a graph-based decoder ([Dozat et Manning, 2018](https://aclanthology.org/P18-2077/)) that take the hidden contextualizations and reconstruct the emotion-cause relations between utterances. 

In order to contextualize utterance text information, we fine-tuned BERT ([Devlin et al., 2019](https://aclanthology.org/N19-1423/)) and used the CLS token as the utterance text contextualization. Let $C=(U_1,...,U_m)$ a conversation of $m$ utterances where each utterance $U_i=(w_1,...,w_{|U_i|})$ is conformed by a sequence of $|U_i|$ words. We passed as input to the encoder the sequence $(b_0,w_1,...,w_{|U_i|} )$ (where $b_0$ is the CLS special token) and used the last hidden states $(\mathbf{h}_0, \mathbf{h}_1, ..., \mathbf{h}_{|U_i|})$ as utterance contextualizations. We conducted some experiments to add image and audio information by jointly contextualizing sequence of frames with audio embeddings. For this purpose, we did some experiments with BEiT ([Bao et al., 2021](https://arxiv.org/abs/2106.08254)) and wav2vec 2.0 ([Baevski et al., 2020](https://arxiv.org/abs/2006.11477)) to concatenate to the text embeddings, the BEiT + wav2vec hidden states. However, due to our resources limitations, we could not properly finetune the complete multimodal encoder architecture (neither try large vision models), so we decided to only participate in the first subtask and report the results of our experiments only in the paper submission.

The decoder design is completely independent of the encoder. Using the encoder outputs $(\mathbf{u}_1,...,\mathbf{u}_m)$ - note here that $\mathbf{u}_i$ represents the utterance $U_i$ and might be a projection of multimodal embeddings or the actual CLS embeddings extracted from BERT - the decoder uses two Biaffine Attention modules: the first directly predicts the scores of the adjacent matrix of the graph while the second predicts the probability of the emotion associated to each causel relation. Since the emotions are only associated to utterance (not causal relations), a post-processing step was needed to predict a single emotion per utterance. 

To incorporate span information we used the words contextualizations returned by BERT and compute a simple attention product between each word sequecence and the utterance embeddings $(\mathbf{u}_1,...,\mathbf{u}_m)$. 



## Implementation details


Our neural architecture was completely designed using Python 3.10 and [Pytorch](https://pytorch.org/). The pretrained models are openly available in [HuggingFace](https://huggingface.co/models). All code will be publicly released with the paper submission and we have plans to maintain it for future work experiments.


