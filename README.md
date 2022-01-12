# Code Documentation for Cross Lingual Summarization
The source directory contains two notebooks and a script file. Each of the notebooks provides the implementation of the methods specified in the report. Specifically `cs626_final_project.ipynb` provides implementation for plain and Multi-Tasking Transformer for Cross lingual Summarization. `Seq2Seq_CLS_Model.ipynb` contains the implementation for the sequence to sequence attention based model. The documentation of `Seq2Seq_CLS_Model.ipynb` is provided in the same notebook. The Script file `data-scrapping.py` contains code for generating the dataset. In each of the notebook files, we have dedicated one cell for installing the dependencies. Thus, in the `requirements.txt` we only mention the dependencies for running the script: `data-scrapping.py`. Please import the drive folder into your personal drive account before running the scripts from this [link](https://drive.google.com/drive/folders/1p-Ve-3JjHRpmw5cDPGqxhpQ8eTJRsT1m?usp=sharing). It contains the datasets necessary for training the models to cross-lingual summarize. Note that, the notebook files can only be run in a Colab environment. In the following subsections we explain the utility of each of the functions present in the aforementioned code files. We have also added sufficient comments to make the code easier to understand.

### cs626_final_project.ipynb

##### `Class FastText(Vectors)`
- `Vectors`: Base class for defining this class. Imported from `torchtext.vocab`

In the code, we have used **FastText word embeddings** for hindi and **Glove word embeddings** for english. This was facilitated by the `TorchText` library. However, the original implementation of accessing FastText word embeddings doesn't work well in the colab environment. Thus, we have mutated the code by creating the aforementioned class which works well under the colab environment.

##### `def english_tokenize(sentence)`
- `sentence` - Input sentence in `str` format

Uses **NLTK Punkt** Tokenizer to tokenize english sentence.

##### `def hindi_tokenize(sentence)`
- `sentence` - Input sentence in `str` format

Uses **Indic-NLP** library for tokenizing the hindi sentence.

##### `def parse_using_torchtext(csv_file_name, batch_size=16, english_vocab_size=10000, hindi_vocab_size=10000)`
- `csv_file_name`: Path to the csv file to be parsed
- `batch_size`: BucketIterator batch size
- `english_vocab_size`: Size of english vocabulary
- `hindi_vocab_size`: Size of hindi vocabulary

This function helps in producing the vocabulary tokenizer from the dataset file corresponding to the `csv_file_name`. The `csv_file_name` is assumed to have a column for english sentences and another one for hindi sentences. After producing the tokenizer for hindi and english language, it parses the dataset by creating a `BucketIterator` necessary for training the networks efficiently.

##### `def parse_using_field(csv_file_name, english_field, hindi_field, english_col_name, hindi_col_name, batch_size=16)`
- `csv_file_name`: Path to the csv file to be parsed
- `english_field`: English Field object that stores the tokenizer and vocabulary for English
- `hindi_field`: Hindi Field object that stores the tokenizer and vocabulary for Hindi
- `english_col_name`: The column in the csv file that contain english sentences
- `hindi_col_name`: The column in the csv file that contain hindi sentences

This function parses the dataset by creating a `BucketIterator` by using a predefined field (tokenizer). While the previous function generates the tokenizer and the iterators for the dataset, this function only generates the iterator for the dataset. This is extremely useful when the networks are to be trained over multiple datasets. In this scenario, only a single field tokenizer must be used for parsing the datasets to produce consistent results.

##### `def parse_dataset(csv_file_name, english_col_name, hindi_col_name, max_num=None)`
- `csv_file_name`: Path to the csv file to be parsed
- `english_col_name`: The column in the csv file that contain english sentences
- `hindi_col_name`: The column in the csv file that contain hindi sentences
- `max_num`: Maximum number of sentences to be parsed

This function parses the dataset from a csv file in the form of list of list of tokens for both the languages. This is useful for testing the model on the sentences derived from the test set. 

##### `def positional_encoding_1d(d_model, length):`
- `d_model`: The embedding dimension of the transformer
- `length` : Length of the sentences in a particular batch

This is useful for producing positional encoding for a sentence. This is used in the forward pass of the transformer.

##### `Class CustomTransformer(nn.Module)`
This class initializes the architecture for transformer based cross lingual summarizer. The code for this architecture is sufficiently commented for better understanding of individual statements invoked in this class.

#### `def train_transformer(model, train_iterator, pad_index, num_epoches=1000, learning_rate=1e-4, save_name=None)`
- `model`: Object of the type `CustomTransformer`
- `train_iterator`: Iterator object from which training tokenized sentences can be fetched
- `pad_index`: Index of `<pad>` in the source vocabulary object
- `num_epoches`: Number of epoches for training
- `learning_rate`: Learning rate for parameters updation
- `save_name`: If this provided, the model would be saved at the respective location after every epoch

We provide a function for training the transformer. This function uses Adam Optimizer for parameters updation. We have tried to increase the efficiency with regards to GPU memory usage by deleting all the variables initialized in the GPU space once they are used.

#### `def produce_output(model, sentence, max_length=100):`
- `model`: Transformer object of either `CustomTransformer` or `MultitaskTransformer` type(explained below)
- `sentence`: Input sentence in `str` or tokenized list format
- `max_length`: Maximum length of the output sentence to be produced.

The transformer takes input in the form of sequence of numbers where each number denotes the word corresponding to its position in the vocabulary. The transformer produces output in the form of numbers as well with no change in the interpretations. This function makes our life easier by letting us input in English and this function automatically converts the transformer output to the target langauge. This function was used for the demo and is used extensively for the validation

##### `def report_performance(model, english_sentences, hindi_sentences)`
- `model`: Transformer object
- `english_sentences`: Sentences in list of list of tokens format
- `hindi_sentences`: Reference Sentences in list of list of tokens format

This reports average bleu score, rouge1 score and rougeL score for the model by using the hindi_sentences as the reference sentence.

##### `class MultitaskTransformer(nn.Module)`
We have implemented a state of the art architecture for cross lingual summarization from scratch. The implementation includes one encoder and several decoders for shared tasks. The theory for this is explained in the report. The code for this architecture can be used for future research.

##### `def train_multitask_transformer(model, mt_iterator, cls_iterator, pad_index, num_epoches=1000, learning_rate=1e-4, save_name=None)`
- `model`: Object of the type `MultitaskTransformer`
- `mt_iterator`: Bucket Iterator which loads sentences from the machine translation corpus
- `cls_iterator`: Bucket Iterator which loads sentences from the cross lingual summarization corpus
- `pad_index`: Index of `<pad>` in the source vocabulary object
- `num_epoches`: Number of epoches for training
- `learning_rate`: Learning rate for parameters updation
- `save_name`: If this provided, the model would be saved at the respective location after every epoch

This is customized for the training the multitask-transformer. Unlike usual sequence to sequence modelling, we have to supply two data loaders for this task:
- Machine Translation Bucket Iterator
- Cross lingual Summarization Bucket Iterator

This training function trains one encoder and two decoders for their respective tasks specializing the encoder weights in understanding the language to a much greater extent.

### data_scrapping.py
This files houses code for parsing the dataset, loading it and also to produce cross lingual corpus.

##### `def generate_data_dictionary(paths, destination='data.pickle')`
- `paths`: list of path in string format which contains sentences to be parsed
- `destination`: Destination path in string format for storing the data in the parsed pickled format

This is used for producing a pickle file which contains the parsed_dataset in a tokenized format.
This is used in one of the functions yet to be described.

##### `def pickle_hindi_data_for_LM(data_dir)`
- `data_dir`: Path to the source directory containing the dataset

This is used for parsing a directory of text files to a suitable format for learning the langauge model for hindi.

##### `def generate_cross_lingual_summary_dataset_new(csv_file_name, target_file_name)`
- `csv_file_name`: Path to mono lingual csv file 
- `target_file_name`: After translation, the csv file is stored at this location.

This uses google translate API for producing a cross lingual summarization dataset from a monolingual summarization dataset.
