'''
@Description: train embedding & tfidf & autoencoder
@FilePath: /bookClassification/src/word2vec/embedding.py
'''
import pandas as pd
from gensim import models
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
from gensim.models import LdaMulticore
from gensim.models.ldamodel import LdaModel
import gensim
from gensim.models.callbacks import CallbackAny2Vec
from gensim.models.word2vec import LineSentence
from gensim.corpora.dictionary import Dictionary
import logging
import os.path

from __init__ import *
from src.utils.config import root_path
from src.utils.tools import create_logger, query_cut, filter_stop_word
from src.word2vec.autoencoder import AutoEncoder

logger = create_logger(root_path + '\\logs\\embedding.log')
# logging.basicConfig(filename=root_path + '\\logs\\embedding_train.log',
#                     format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


class callback(CallbackAny2Vec):
    '''Callback to print loss after each epoch.'''

    def __init__(self):
        self.epoch = 0
        self.loss_to_be_subed = 0

    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        loss_now = loss - self.loss_to_be_subed
        self.loss_to_be_subed = loss
        print('Loss after epoch {}: {}'.format(self.epoch, loss_now))
        logging.info('Loss after epoch {}: {}'.format(self.epoch, loss_now))
        self.epoch += 1


class SingletonMetaclass(type):
    '''
    @description: singleton
    '''
    def __init__(self, *args, **kwargs):
        self.__instance = None
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        if self.__instance is None:
            self.__instance = super(SingletonMetaclass,
                                    self).__call__(*args, **kwargs)
            return self.__instance
        else:
            return self.__instance


class Embedding(metaclass=SingletonMetaclass):
    def __init__(self):
        '''
        @description: This is embedding class. Maybe call so many times. we need use singleton model.
        In this class, we can use tfidf, word2vec, fasttext, autoencoder word embedding
        @param {type} None
        @return: None
        '''
        # 停止词
        self.stopWords = [
            x.strip('\n') for x in open(root_path + '\\data\\stopwords.txt', encoding='utf-8').readlines()
        ]
        # print("stopword", self.stopWords)
        ### self.stopWords = open(root_path + '/data/stopwords.txt', encoding='utf-8').readlines()
        # autuencoder
        self.ae = AutoEncoder()      #这部分训练完词向量再看

    def load_data(self):
        '''
        @description:Load all data, then do word segment
        @param {type} None
        @return:None
        '''
        logger.info('load data')
        if not os.path.isfile(root_path + '\\data\\embedding_corpus.txt'):
            self.data = pd.concat([
                pd.read_csv(root_path + '\\data\\train.csv', sep='\t'),
                pd.read_csv(root_path + '\\data\\dev.csv', sep='\t'),
                pd.read_csv(root_path + '\\data\\test.csv', sep='\t')
            ])
            self.data["text"] = self.data['title'] + self.data['desc']
            self.data["text"] = self.data["text"].apply(query_cut)
            self.data["text"] = self.data["text"].apply(filter_stop_word)
            self.data['text'] = self.data["text"].apply(lambda x: " ".join(x))

            with open(root_path + '\\data\\embedding_corpus.txt', 'w', encoding='utf-8') as f:
                for senten in self.data['text']:
                    f.write(senten+"\n")

        with open(root_path + '\\data\\embedding_corpus.txt', encoding='utf-8') as f:
            self.fit_data = f.readlines()
        self.fit_data = [x.strip("\n") for x in self.fit_data]
        self.lda_data = [x.split(" ") for x in self.fit_data]

    def trainer(self):
        '''
        @description: Train tfidf,  word2vec, fasttext and autoencoder
        @param {type} None
        @return: None
        '''
        # logger.info('train tfidf')
        # count_vect = TfidfVectorizer(stop_words=self.stopWords,
        #                              max_df=0.4,
        #                              min_df=0.001,
        #                              ngram_range=(1, 2))
        # self.tfidf = count_vect.fit(self.fit_data)


        # logger.info("running %s" % ' '.join(sys.argv))
        # # self.data['text'] = self.data["text"].apply(lambda x: x.split(' '))
        # logger.info('train word2vec')
        # self.w2v = models.Word2Vec(min_count=2,
        #                            window=5,
        #                            size=300,
        #                            sample=6e-5,
        #                            alpha=0.03,
        #                            min_alpha=0.0007,
        #                            negative=15,
        #                            workers=4,
        #                            iter=40,
        #                            max_vocab_size=50000)
        # self.w2v.build_vocab(LineSentence(root_path + '\\data\\embedding_corpus.txt'))
        # self.w2v.train(LineSentence(root_path + '\\data\\embedding_corpus.txt'),
        #                total_examples=self.w2v.corpus_count,
        #                epochs=self.w2v.iter,
        #                compute_loss=True,
        #                callbacks=[callback()])


        logger.info('train fast')
        #训练fast的词向量
        self.fast = models.FastText(
            size=300,  # 向量维度
            window=3,  # 移动窗口
            alpha=0.03,
            min_count=2,  # 对字典进行截断, 小于该数的则会被切掉,增大该值可以减少词表个数
            iter=40,  # 迭代次数
            min_n=1,
            max_n=4,
            word_ngrams=1,
            max_vocab_size=50000)
        self.fast.build_vocab(LineSentence(root_path + '\\data\\embedding_corpus.txt'))
        self.fast.train(LineSentence(root_path + '\\data\\embedding_corpus.txt'),
                        total_examples=self.fast.corpus_count,
                        epochs=self.fast.iter)

        # self.fast = models.FastText(
        #     self.data["text"],
        #     size=300,  # 向量维度
        #     window=3,  # 移动窗口
        #     alpha=0.03,
        #     min_count=2,  # 对字典进行截断, 小于该数的则会被切掉,增大该值可以减少词表个数
        #     iter=30,  # 迭代次数
        #     max_n=3,
        #     word_ngrams=2,
        #     max_vocab_size=50000)

        # logger.info('train lda')
        # self.id2word = Dictionary(self.lda_data)
        # corpus = [self.id2word.doc2bow(text) for text in self.lda_data]
        # self.LDAmodel = LdaMulticore(corpus=corpus,
        #                              id2word=self.id2word,
        #                              num_topics=30,
        #                              workers=4,
        #                              chunksize=4000,
        #                              passes=7,
        #                              alpha='asymmetric')

        # logger.info('train autoencoder')
        # self.ae.train(self.fit_data)

    def saver(self):
        '''
        @description: save all model
        @param {type} None
        @return: None
        '''
        # logger.info('save autoencoder model')
        # self.ae.save()

        # logger.info('save tfidf model')
        # joblib.dump(self.tfidf, root_path + '\\model\\embedding/tfidf')

        # logger.info('save w2v model')
        # self.w2v.wv.save_word2vec_format(root_path +
        #                                  '\\model\\embedding\\w2v.bin',
        #                                  binary=False)

        # logger.info('save fast model')
        # self.fast.wv.save_word2vec_format(root_path +
        #                                   '\\model\\embedding\\fast.bin',
        #                                   binary=False)

        logger.info('save complete fast model')
        self.fast.save(root_path +'\\model\\embedding\\complete\\fastmodel')

        # logger.info('save lda model')
        # self.LDAmodel.save(root_path + '/model/embedding/lda')

    def load(self):
        '''
        @description: Load all embedding model
        @param {type} None
        @return: None
        '''
        logger.info('load tfidf model')
        self.tfidf = joblib.load(root_path + '/model/embedding/tfidf')
        #
        logger.info('load w2v model')
        self.w2v = models.KeyedVectors.load_word2vec_format(
            root_path + '/model/embedding/w2v.bin', binary=False)

        logger.info('load fast model')
        self.fast = models.KeyedVectors.load_word2vec_format(
            root_path + '/model/embedding/fast.bin', binary=False)

        logger.info('load complete fast model')
        self.fastmodel = models.FastText.load(root_path + '/model/embedding/complete/fastmodel')

        logger.info('load lda model')
        self.lda = LdaModel.load(root_path + '/model/embedding/lda')

        logger.info('load autoencoder model')
        self.ae.load()


if __name__ == "__main__":
    em = Embedding()
    em.load_data()
    em.trainer()
    em.saver()
    # em.load()
    # em.lda.print_topics(num_topics=-1, num_words=10)
    # document = em.lda.id2word.doc2bow(["你", "今天", "吃", "核桃"])
    # print(em.lda.get_document_topics(document, minimum_probability=0))
    # # print(len(em.w2v["校园"]))
    # print(em.fast.most_similar("演员"))