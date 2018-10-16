---
title: MedNLI — A Natural Language Inference Dataset For The Clinical Domain
---

## MedNLI

Natural Language Inference (NLI) is one of the critical tasks for understanding natural language. The objective of NLI is to determine if a given hypothesis can be inferred from a given premise. NLI systems have made significant progress over the years, and has gained popularity since the recent release of datasets such as the Stanford Natural Language Inference (SNLI) (Bowman et al. 2015) and Multi-NLI (Nangia et al. 2017).

We introduce MedNLI - a dataset annotated by doctors, performing a natural language inference task), grounded in the medical history of patients. We present strategies to: 1) leverage transfer learning using datasets from the open domain, (e.g. SNLI) and 2) incorporate domain knowledge from external data and lexical sources (e.g. medical terminologies). Our results demonstrate performance gains using both strategies.

### Access the data
We make MedNLI available through the MIMIC-III derived data repository. Any individual certified to access MIMIC-III can access MedNLI.

[http://doi.org/10.13026/C2RS98](http://doi.org/10.13026/C2RS98)

### Code
Code to reproduce the results: [https://github.com/jgc128/mednli](https://github.com/jgc128/mednli)

A simple ready-to-use baseline with pre-trained models: [https://github.com/jgc128/mednli_baseline](https://github.com/jgc128/mednli_baseline)

## Reference
The paper was accepted to EMNLP 2018! Meanwhile, here is an extended arXiv version:

Romanov, A., & Shivade, C. (2018). Lessons from Natural Language Inference in the Clinical Domain. arXiv preprint arXiv:1808.06752.  
[https://arxiv.org/abs/1808.06752](https://arxiv.org/abs/1808.06752)


```
@article{romanov2018lessons,
	title = {Lessons from Natural Language Inference in the Clinical Domain},
	url = {http://arxiv.org/abs/1808.06752},
	abstract = {State of the art models using deep neural networks have become very good in learning an accurate mapping from inputs to outputs. However, they still lack generalization capabilities in conditions that differ from the ones encountered during training. This is even more challenging in specialized, and knowledge intensive domains, where training data is limited. To address this gap, we introduce {MedNLI} - a dataset annotated by doctors, performing a natural language inference task ({NLI}), grounded in the medical history of patients. We present strategies to: 1) leverage transfer learning using datasets from the open domain, (e.g. {SNLI}) and 2) incorporate domain knowledge from external data and lexical sources (e.g. medical terminologies). Our results demonstrate performance gains using both strategies.},
	journaltitle = {arXiv:1808.06752 [cs]},
	author = {Romanov, Alexey and Shivade, Chaitanya},
	urldate = {2018-08-27},
	date = {2018-08-21},
	eprinttype = {arxiv},
	eprint = {1808.06752},
}
```
