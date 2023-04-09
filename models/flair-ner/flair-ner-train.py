from flair.data import Corpus
from flair.datasets import ColumnCorpus
from flair.embeddings import WordEmbeddings, FlairEmbeddings, StackedEmbeddings, TokenEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
from typing import List

class FlairNERTrain:
    def __init__(self):
        self.columns = {0 : 'text', 1 : 'ner'}
        self.data_folder = 'corpus/'
        self.label_type = 'ner'
        self.embedding_types : List[TokenEmbeddings] = [
            WordEmbeddings('glove'),
            FlairEmbeddings('news-forward'),
            FlairEmbeddings('news-backward'),
        ]
        self.embeddings : StackedEmbeddings = StackedEmbeddings(
            embeddings=self.embedding_types)
        self.tagger : SequenceTagger = None
        self.trainer : ModelTrainer = None
    
    def train(self, train_file, test_file, dev_file, tagger_path):
        # initializing the corpus
        corpus: Corpus = ColumnCorpus(self.data_folder, self.columns,
                                      train_file=train_file,
                                      test_file=test_file,
                                      dev_file=dev_file)
        
        # make tag dictionary from the corpus
        label__dictionary = corpus.make_label_dictionary(label_type=self.label_type)

        self.tagger = SequenceTagger(hidden_size=256,
                                     embeddings=self.embeddings,
                                     tag_dictionary=label__dictionary,
                                     tag_type=self.label_type,
                                     use_crf=True)

        self.trainer = ModelTrainer(self.tagger, corpus)

        self.trainer.train(tagger_path,
                    learning_rate=0.1,
                    mini_batch_size=32,
                    max_epochs=25)

def main(just_run: bool = False):
    ner_trainer = FlairNERTrain()
    ner_trainer.train(train_file='train.txt',
                      test_file='test.txt',
                      dev_file='dev.txt',
                      tagger_path='model-info/')


