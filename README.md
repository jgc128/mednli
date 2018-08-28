MedNLI - Natural Language Inference in Clinical Texts
=====================================================

## Installation

1. Clone this repo: `git clone ...`
2. Install requirements: `pip install requirements.txt`
3. Install PyTorch v0.2.0: `pip isntall http://download.pytorch.org/whl/cu80/torch-0.2.0.post3-cp36-cp36m-manylinux1_x86_64.whl` (see https://github.com/pytorch/pytorch#installation for details)
4. Install MetaMap: https://metamap.nlm.nih.gov/Installation.shtml
   - Make sure to set `METAMAP_BINARY_PATH` in the `config.py` to your MetaMap binary installation
5. Install PyMetaMap: https://github.com/AnthonyMRios/pymetamap
6. Install UMLS Metathesaurus: https://www.nlm.nih.gov/research/umls/
   - Make sure to set `UMLS_INSTALLATION_DIR` in the `config.py` pointing to your UMLS installation 


## Downloading the datasets

1. Download SNLI: https://nlp.stanford.edu/projects/snli/
2. Download MultiNLI: http://www.nyu.edu/projects/bowman/multinli/ (we experimented with MultiNLI v0.9)
3. Download MedNLI: https://jgc128.github.io/mednli/

Put all of the data inside the `./data/` dir so is has the following structure:
```
$ ls data/
mednli_1.0  multinli_0.9  snli_1.0
``` 

```
$ ls data/snli_1.0/
README.txt  snli_1.0_dev.jsonl  snli_1.0_dev.txt  snli_1.0_test.jsonl  snli_1.0_test.txt  snli_1.0_train.jsonl  snli_1.0_train.txt
```

## Downloading the word embeddings

| Word Embedding  | Link |
| ------------- | ------------- |
|glove |  [glove.840B.300d.pickled](https://mednli.blob.core.windows.net/shared/word_embeddings/glove.840B.300d.pickled) |
|mimic |  [mimic.fastText.no_clean.300d.pickled](https://mednli.blob.core.windows.net/shared/word_embeddings/mimic.fastText.no_clean.300d.pickled) |
|bio_asq | [bio_asq.no_clean.300d.pickled](https://mednli.blob.core.windows.net/shared/word_embeddings/bio_asq.no_clean.300d.pickled) |
|wiki_en | [wiki_en.fastText.300d.pickled](https://mednli.blob.core.windows.net/shared/word_embeddings/wiki_en.fastText.300d.pickled) |
|wiki_en_mimic |  [wiki_en_mimic.fastText.no_clean.300d.pickled](https://mednli.blob.core.windows.net/shared/word_embeddings/wiki_en_mimic.fastText.no_clean.300d.pickled) |
|glove_bio_asq |  [glove_bio_asq.no_clean.300d.pickled](https://mednli.blob.core.windows.net/shared/word_embeddings/glove_bio_asq.no_clean.300d.pickled) |
|glove_bio_asq_mimic |[glove_bio_asq_mimic.no_clean.300d.pickled](https://mednli.blob.core.windows.net/shared/word_embeddings/glove_bio_asq_mimic.no_clean.300d.pickled) |

Put all embeddings inside the `./data/word_embeddings/` dir so is has the following structure:

```
$ ls data/word_embeddings/
glove.840B.300d.pickled		glove_bio_asq_mimic.no_clean.300d.pickled 	mimic.fastText.no_clean.300d.pickled
```


## Running the code
Code tested on Python 3.4 and Python 3.6.3

0. Configuration: `config.py`
1. Preprocess the data: `python preprocess.py`
   - This script will create files `genre_*.pkl` in the `./data/nli_processed/` directory
   - Preprocess the test data: `python preprocess.py process_test`
2. Extract concepts: `python metamap_extract_concepts.py`
   - Make sure to run MetaMap servers first before executing this script 
   - The script above works only for the MedNLI dataset. Rename the files `genre_*.pkl` to `genre_concepts_*.pkl` for SNLI and all MultiNLI domains.
   - Call `main_data_test` as the main function to process the test data
3. Create word embeddings cache: `python pickle_word_vectors.py <path_to_glove/word2vec file> ./data/word_embeddings/<name>`
   - See `WORD_VECTORS_FILENAME` in the `config.py` for file namings
4. Create UMLS graph cache: `python parse_umls_create_concepts_graph.py`
5. Optional: to create input data for the [official retrofitting script](https://github.com/mfaruqui/retrofitting) run `python create_retorfitting_data.py`
6. Train the model: `python train_model.py`
   - You can change the parameters in the `config` function or in the command line: `python train_model.py with use_umls_attention=True use_token_level_attention=True` (see the [Sacred documentation](http://sacred.readthedocs.io/en/latest/) for details)
 

### Configuration options
```python
model_class = 'PyTorchInferSentModel' # class name of the model to run. See the `create_model` function for the available models
max_len = 50 # max sentence length
lowercase = False # lowercase input data or nor
clean = False # remove punctuation etc or not
stem = False # do stemming to not
word_vectors_type = 'glove'  # word vectors - see the `WORD_VECTORS_FILENAME` in `config.py` for details
word_vectors_replace_cui = ''  # filename with retorifitted embeddings for CUIs, eg cui.glove.cbow_most_common.CHD-PAR.SNOMEDCT_US.retrofitted.pkl
downsample_source = 0 # down sample the source domain data to the size of the MedNLI

# transfer learning settings
genre_source = 'clinical' # source domain for transfer learning. target='' and tune='' - no transfer
genre_target = '' # target domain - always MedNLI in case of experiemnts in the paper
genre_tune = '' # fine-tuning domain
lambda_multi_task = -1 # whether to use dynamically sampled batches from different domains or not.
uniform_batches = True # a batch will contain samples from just one domain

rnn_size = 300 # size of LSTM
rnn_cell = 'LSTM' # LSTM is used in the experiments in the paper
regularization = 0.000001 # regularization strength
dropout = 0.5 # dropout
hidden_size = 300 # size of the hidden fully-connected layers
trainable_embeddings = False # train embeddings or not

# knowledge-based attention
# set both to true to reproduce the token-level UMLS attention used in the paper
use_umls_attention = False # whether to use the knowledge-based attention or not
use_token_level_attention = False # use CUIs or separate tokens for attention

batch_size = 512 # batch size
epochs = 40 # number of epochs for training
learning_rate = 0.001 # learning rate for the Adam optimizer
training_loop_mode = 'best_loss'  # best_loss or best_acc - the model will be saved on the base loss or accuracy on the validation set correspondingly

```


## Experiments in the paper

### Baselines
To run the BOW, InferSent, and ESIM models with default settings, use the following commands:

```
python train_model.py with model_class=PyTorchSimpleModel
python train_model.py with model_class=PyTorchInferSentModel
python train_model.py with model_class=PyTorchESIMModel
```

### Transfer learning
To pre-train the model on the `Slate` domain, fine-tune on the MedNLI and test on the dev set of MedNLI (Sequential transfer in the paper), run the following command:

`python train_model.py with genre_source=slate genre_tune=clinical genre_target=clinical`

To run the Multi-target transfer learning, specify the genres and use the corresponding versions of the models: `PyTorchMultiTargetSimpleModel`, `PyTorchMultiTargetInferSentModel`, and `PyTorchMultiTargetESIMModel`.


### Word embeddings
All word embeddings have to be pickled first - see the `pickle_word_embeddings.py` script.
To run the model with a specific embeddings, use the `word_vectors_type` parameter:

`python train_model.py with word_vectors_type=wiki_en_mimic`

### Retorfitting

 - First, create the input data for retrofitting with the `create_retrofitting_data.py` script. 
 - Second, run the official script from GitHub. (https://github.com/mfaruqui/retrofitting).
 - Next, pickle the resulting word vectors with the `pickle_word_vectors.py` script.
 - Finally, set the `word_vectors_replace_cui` parameter to the pickled retrofitted vectors:
   - `python train_model.py with word_vectors_replace_cui=cui.glove.cbow_most_common.CHD-PAR.SNOMEDCT_US.retrofitted.pkl`
   
   
### Knowledge-directed attention
Set both `use_umls_attention` and `use_token_level_attention` to `True` to reproduce the token-level UMLS attention experiments:

`python train_model.py with use_umls_attention=True use_token_level_attention=True`


# Reference
The paper was accepted to EMNLP 2018! Meanwhile, here is an extended arXiv version:

Romanov, A., & Shivade, C. (2018). Lessons from Natural Language Inference in the Clinical Domain. arXiv preprint arXiv:1808.06752.  
https://arxiv.org/abs/1808.06752


```
@article{romanov2018lessons,
	title = {Lessons from Natural Language Inference in the Clinical Domain},
	url = {http://arxiv.org/abs/1808.06752},
	abstract = {State of the art models using deep neural networks have become very good in learning an accurate mapping from inputs to outputs. However, they still lack generalization capabilities in conditions that differ from the ones encountered during training. This is even more challenging in specialized, and knowledge intensive domains, where training data is limited. To address this gap, we introduce {MedNLI} - a dataset annotated by doctors, performing a natural language inference task ({NLI}), grounded in the medical history of patients. We present strategies to: 1) leverage transfer learning using datasets from the open domain, (e.g. {SNLI}) and 2) incorporate domain knowledge from external data and lexical sources (e.g. medical terminologies). Our results demonstrate performance gains using both strategies.},
	journaltitle = {{arXiv}:1808.06752 [cs]},
	author = {Romanov, Alexey and Shivade, Chaitanya},
	urldate = {2018-08-27},
	date = {2018-08-21},
	eprinttype = {arxiv},
	eprint = {1808.06752},
}
```
