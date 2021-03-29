import pandas as pd
import numpy as np
import pickle
import re
import os
import gensim
import spacy
import el_core_news_md
nlp=el_core_news_md.load()


#-----------------------------------------Functions-----------------------------------------------#	
#----------------------------------------------------------------------------------------------------#
class ProgressBar(object):
    DEFAULT = 'Progress: %(bar)s %(percent)3d%%'
    FULL = '%(bar)s %(current)d/%(total)d (%(percent)3d%%) %(remaining)d to go'

    def __init__(self, total, width=40, fmt=DEFAULT, symbol='=',
                 output=sys.stderr):
        assert len(symbol) == 1

        self.total = total
        self.width = width
        self.symbol = symbol
        self.output = output
        self.fmt = re.sub(r'(?P<name>%\(.+?\))d',
            r'\g<name>%dd' % len(str(total)), fmt)

        self.current = 0

    def __call__(self):
        percent = self.current / float(self.total)
        size = int(self.width * percent)
        remaining = self.total - self.current
        bar = '[' + self.symbol * size + ' ' * (self.width - size) + ']'

        args = {
            'total': self.total,
            'bar': bar,
            'current': self.current,
            'percent': percent * 100,
            'remaining': remaining
        }
        print('\r' + self.fmt % args, file=self.output, end='')

    def done(self):
        self.current = self.total
        self()
        print('', file=self.output)
		
# _____TF-IDF libraries_____
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# _____helper Libraries_____
import pickle  # would be used for saving temp files
import csv     # used for accessing the dataset
import timeit  # to measure time of training
import random  # used to get a random number


def tf_idf_generate(sentences):
    #https://stackoverflow.com/questions/30976120/find-the-tf-idf-score-of-specific-words-in-documents-using-sklearn

    from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
    # our corpus
    data = sentences

    cv = CountVectorizer()

    # convert text data into term-frequency matrix
    data = cv.fit_transform(data)

    tfidf_transformer = TfidfTransformer()

    # convert term-frequency matrix into tf-idf
    tfidf_matrix = tfidf_transformer.fit_transform(data)

    # create dictionary to find a tfidf word each word
    word2tfidf = dict(zip(cv.get_feature_names(), tfidf_transformer.idf_))
    return word2tfidf

#https://stackoverflow.com/questions/38088652/pandas-convert-categories-to-numbers
from nltk import Tree
#from nltk.tokenize import word_tokenize
#from nltk.tag import pos_tag

#ex = 'European authorities fined Google a record $5.1 billion on Wednesday for abusing its power in the mobile phone market and ordered the company to alter its practices'

def get_pos_tags_dict(word_dict):

    pos_list ={}
    for word in word_dict.keys():
        doc=nlp(word)
        for token in doc:
            pos_list[word] = token.pos_
    print(pos_list)

    import pandas as pd
    df = pd.DataFrame(list(pos_list.items()))
    df.columns = ['word', 'pos']
    df.pos = pd.Categorical(df.pos)
    df['code'] = df.pos.cat.codes
    print(df)

    pos_list ={}
    for index, row in df.iterrows():
       pos_list[row['word']] = row['code']
    print(pos_list)
    return pos_list 

#https://stackoverflow.com/questions/38088652/pandas-convert-categories-to-numbers
from nltk import Tree
#from nltk.tokenize import word_tokenize
#from nltk.tag import pos_tag

#ex = 'European authorities fined Google a record $5.1 billion on Wednesday for abusing its power in the mobile phone market and ordered the company to alter its practices'

def get_dep_tags_dict(sentence_list):

    dep_list ={}
    for sentence in sentence_list:
        doc=nlp(sentence)
        for word in doc:
            dep_list[word] = word.dep_
            print(word)
            print(word.dep_)
    #print(dep_list)

    import pandas as pd
    df = pd.DataFrame(list(dep_list.items()))
    df.columns = ['word', 'dep']
    df.dep = pd.Categorical(df.dep)
    df['code'] = df.dep.cat.codes
    deps=df["code"]
    #print(df)

    dep_list ={}
    for index, row in df.iterrows():
       dep_list[row['word']] = row['code']
       #print(row['word'])
       #print(row['code'])
    #print(dep_list)
    return dep_list,deps 

def get_init_embedding(model, text_list, summary_list, embedding_size):
    print("Loading Lists...")
    #train_article_list = get_text_list(train_article_path, False)
    #train_title_list = get_text_list(train_title_path, False)

    print("Loading TF-IDF...")
    tf_idf_list = tf_idf_generate(text_list+summary_list)
    
    print("Loading Pos Tags...")
    pos_list = get_pos_tags_dict(model.wv.vocab)

    print("Loading Dep Tags...")
    dep_list = get_dep_tags_dict(text_list[:1000])

    
    #print("Loading Named Entity...")
    #named_entity_recs = named_entity(postags_for_named_entity) 
    
    #print("Loading Glove vectors...")

    #with open( default_path + "glove/model_glove_300.pkl", 'rb') as handle:
     #   word_vectors = pickle.load(handle)
     
    
    used_words = 0
    word_vec_list = list()
    word_embs ={}
    for word in sorted(model.wv.vocab.keys()):
        try:
            #word_vec = word_vectors.word_vec(word)
            word_vec = model.wv[word]
            if word in tf_idf_list:
              v= tf_idf_list[word]
              print(v)
              rich_feature_array = np.array([v,v,v,v,v,v,v,v,v,v])
              word_vec = np.append(word_vec, rich_feature_array)
              word_embs[word]=word_vec
            else:
              v=0
              rich_feature_array = np.array([v,v,v,v,v,v,v,v,v,v])
              word_vec = np.append(word_vec, rich_feature_array)
              word_embs[word]=word_vec

            if word in pos_list:
              v=pos_list[word]
              print(v)
              rich_feature_array_2 = np.array([v,v,v,v,v,v,v,v,v,v])
              word_vec = np.append(word_vec, rich_feature_array_2)
              word_embs[word]=word_vec
            else:
              v=0
              rich_feature_array_2 = np.array([v,v,v,v,v,v,v,v,v,v])
              word_vec = np.append(word_vec, rich_feature_array_2) 
              word_embs[word]=word_vec

            if word in dep_list:
              v=dep_list[word]
              print(v)
              rich_feature_array_3 = np.array([v,v,v,v,v,v,v,v,v,v])
              word_vec = np.append(word_vec, rich_feature_array_3)
              word_embs[word]=word_vec
            else:
              v=0
              rich_feature_array_3 = np.array([v,v,v,v,v,v,v,v,v,v])
              word_vec = np.append(word_vec, rich_feature_array_3)  
              word_embs[word]=word_vec
          
            used_words += 1
        except KeyError:
            word_vec = np.zeros([embedding_size], dtype=np.float32) #to generate for <padding> and <unk>
            word_embs[word]=word_vec
        
        
        word_vec_list.append(np.array(word_vec))

    print("words found in glove percentage = " + str((used_words/len(word_vec_list))*100) )
          
    return word_embs, np.array(word_vec_list)

def my_save_word2vec_format(fname, vocab, vectors, binary, total_vec):
    """Store the input-hidden weight matrix in the same format used by the original
    C word2vec-tool, for compatibility.

    Parameters
    ----------
    fname : str
        The file path used to save the vectors in.
    vocab : dict
        The vocabulary of words.
    vectors : numpy.array
        The vectors to be stored.
    binary : bool, optional
        If True, the data wil be saved in binary word2vec format, else it will be saved in plain text.
        total_vec : int, optional
    Explicitly specify total number of vectors
    (in case word vectors are appended with document vectors afterwards).
    """

    if not (vocab or vectors):
        raise RuntimeError("no input")
    if total_vec is None:
        total_vec = len(vocab)
    vector_size = vectors.shape[1]
    assert (len(vocab), vector_size) == vectors.shape
    with utils.smart_open(fname, 'wb') as fout:
        print(total_vec, vector_size)
        fout.write(utils.to_utf8("%s %s\n" % (total_vec, vector_size)))
        # store in sorted order: most frequent words at the top
        for word, row in vocab.items():
            if binary:
                row = row.astype(REAL)
                fout.write(utils.to_utf8(word) + b" " + row.tostring())
            else:
                fout.write(utils.to_utf8("%s %s\n" % (word, ' '.join(repr(val) for val in row))))

				
	
def save(obj , filename):
  print("saving {} ..".format(filename))
  with open(filename, 'wb') as handle:
      pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)
      
def load(filename):
  print("loading {} ..".format(filename))
  with open(filename, 'rb') as handle:
    return pickle.load(handle)
	
def final_clean(raw_text):
    #Remove non Greek words
    text=re.sub("[a-zA-Z]",'',raw_text)
    #Remove numbers ?
    text=re.sub("[0-9]",'',text)
    #Remove text in parenthesis 
    text=re.sub(r'\([^)]*\)', '', text)
    #Remove special characters 
    text=re.sub("[\\\]",' ',text)
    text=re.sub("[\^\\/\{\}\[\]!%@#$&\*\-_<>–?:\"\.<>”“+=:;\,«»…]",'',text)
    text=re.sub(r'\’','',text)
    text=re.sub(r'\'','',text)
    #Remove multiple spaces
    text = re.sub("(\s+)",' ',text)
    text = re.sub("(\t)",'',text)
    return text


#-----------------------------------------Main program-----------------------------------------------#	
#----------------------------------------------------------------------------------------------------#
file_data ="data/clean_greek_data_final_vf.csv"
reviews = pd.read_csv(file_data,skiprows=[1],skip_blank_lines=True)
text = reviews["Text"] 
summary = reviews["Summary"]	

clean_text = []
#text_list=[]
print("loading documents...")
progress = ProgressBar(len(text), fmt=ProgressBar.FULL)
for doc in text:
  doc=final_clean(doc)
  clean_text.append(str(doc).split())
  #text_list=text_list.append(doc)
  progress.current += 1
  progress()
progress.done()
save(clean_text , "clean_text.pkl")

clean_summary = []
#summary_list =[]
print("loading summaries...")
progress = ProgressBar(len(summary), fmt=ProgressBar.FULL)
for doc in summary:
  doc=final_clean(doc)
  clean_summary.append(str(doc).split())
  #summary_list=summary_list.append(doc)
  progress.current += 1
  progress()
progress.done()
save(clean_summary , "clean_summary.pkl")

documents_list = clean_text +clean_summary
print("The number of documents is :")
print(len(documents_list))
path="data/word_embeddings"
model_greek_vec = gensim.models.Word2Vec(
        documents_list,
        size=120,
        window=10,
        min_count=2,
        workers=10)
model_greek_vec.train(documents_list, total_examples=len(documents_list), epochs=10)
model_greek_vec.wv.save(path +"my_word_embs.model")
from gensim.models import KeyedVectors
model = KeyedVectors.load(path +"my_word_embs.model", mmap='r')

title=[]
text=[]
summary=[]
title_list=[]
text_list=[]
summary_list=[]
file_data ="data/clean_greek_data_final_vf.csv"
#Read the desired columns
title=pd.read_csv(file_data,skiprows=[1],usecols=["Title"], skip_blank_lines=True)
print(title.info())
text=pd.read_csv(file_data,skiprows=[1],usecols=["Text"], skip_blank_lines=True)
print(text.info())
summary=pd.read_csv(file_data,skiprows=[1],usecols=["Summary"], skip_blank_lines=True)
print(summary.info())

text_list=[]
summary_list=[]
for index,rows in text.iterrows():
    #print(index)
    rows = final_clean(str(rows))
    text_list.append(rows)

for index,rows in summary.iterrows():
    #print(index)
    rows = final_clean(str(rows))
    summary_list.append(rows)

lists=text_list+summary_list
print("The number of documents:")
print(len(lists))

word_embs={}
word_vecs_array=list()
word_embs,word_vecs_array=get_init_embedding(model,text_list, summary_list, 150)

from numpy import zeros, dtype, float32 as REAL, ascontiguousarray, fromstring
from gensim import utils

m = gensim.models.keyedvectors.Word2VecKeyedVectors(vector_size=150)
m.vocab = word_embs
m.vectors = np.array(list(word_embs.values()))
my_save_word2vec_format(binary=False, fname=path+'my_word_embs_feats.bin', total_vec=len(word_embs), vocab=m.vocab, vectors=m.vectors)