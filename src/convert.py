#!pip install ProgressBar2
#import ProgressBar2 

import sys
import os
import time
import hashlib
import struct
import subprocess
import collections
import tensorflow as tf
from tensorflow.core.example import example_pb2
import nltk
#nltk.download('punkt')
#from nltk.tokenize import word_tokenize
import pandas as pd


dm_single_close_quote = u'\u2019' # unicode
dm_double_close_quote = u'\u201d'
END_TOKENS = ['.', '!', '?', '...', "'", "`", '"', dm_single_close_quote, dm_double_close_quote, ")"] # acceptable ways to end a sentence

# We use these to separate the summary sentences in the .bin datafiles
SENTENCE_START = '<s>'
SENTENCE_END = '</s>'

all_train_urls = ""
all_val_urls = ""
all_test_urls = ""

import re
#Clean final data
def final_clean(raw_text):
  #Remove non Greek words
  text=re.sub("[a-zA-Z]",'',raw_text)
  #Remove numbers ?
  text=re.sub("[0-9]",'',text)
  #Remove text in parenthesis 
  text=re.sub(r'\([^)]*\)', '', text)
  #Remove special characters 
  text=re.sub("[\\\]",' ',text)
  text=re.sub("[\^\\/\{\}\[\]!%__@#$&\*\-_<>–?:\"\.<>”“+=:;\,«»…]",'',text)
  text=re.sub(r'\’','',text)
  text=re.sub(r'\'','',text)
  #Remove multiple spaces
  text = re.sub("(\s+)",' ',text)
  text = re.sub("(\t)",'',text)
  return text


def fix_missing_period(line):
  """Adds a period to a line that is missing a period"""
  if "@highlight" in line: return line
  if line=="": return line
  if line[-1] in END_TOKENS: return line
  # print line[-1]
  return line 

def read_text_file(text_file):
  lines = []
  with open(text_file, "r") as f:
    for line in f:
      lines.append(line.strip())
  return lines

def get_art_abs(story_file):
  lines = read_text_file(story_file)

  # Lowercase everything
  lines = [line.lower() for line in lines]

  # Put periods on the ends of lines that are missing them (this is a problem in the dataset because many image captions don't end in periods; consequently they end up in the body of the article as run-on sentences)
  lines = [fix_missing_period(line) for line in lines]
  #print(lines)

  # Separate out article and abstract sentences
  article_lines = []
  highlights = []
  next_is_highlight = False
  for idx,line in enumerate(lines):
    if line == "":
      continue # empty line
    elif line.startswith("@highlight"):
      next_is_highlight = True
    elif next_is_highlight:
      highlights.append(line)
    else:
      article_lines.append(line)

  # Make article into a single string
  article = ' '.join(article_lines)
  #print(article)

  return article

def write_to_bin(file_name):
#def write_to_bin(article):
  article = get_art_abs(file_name)
  article = final_clean(article)
  #out_file = 'my_test' + time.strftime("%Y%m%d-%H%M%S") + '.bin'
  out_file = file_name[:len(file_name)-4] + time.strftime("%Y%m%d-%H%M%S") + '.bin'
  # Write to tf.Example
  with open(out_file, 'wb') as writer:
    tf_example = example_pb2.Example()
    tf_example.features.feature['article'].bytes_list.value.extend([article.encode('utf-8')])
    #tf_example.features.feature['abstract'].bytes_list.value.extend([abstract.encode('utf-8')])
    tf_example_str = tf_example.SerializeToString()
    str_len = len(tf_example_str)
    writer.write(struct.pack('q', str_len))
    writer.write(struct.pack('%ds' % str_len, tf_example_str))
  return out_file
#write_to_bin('myarticle.txt')
#article = "Ενδιαμέσως η Αθήνα έχει αποδυθεί σε ένα πολύ προσεκτικό διπλωματικό πόκερ με το Κάιρο, σε μια προσπάθεια να ολοκληρωθεί η συμφωνία οριοθέτησης ΑΟΖ. Αγκάθι παραμένει το Καστελλόριζο, το οποίο η Αίγυπτος δεν επιθυμεί να αποτελεί μέρος των διαπραγματεύσεων. Ο Ιούλιος κρίνεται ιδιαίτερα σημαντικός για την πορεία αυτών των συνομιλιών, οι οποίες δεν αποκλείεται να συνεχιστούν και τις πρώτες ημέρες του Αυγούστου, έτσι ώστε μέσα στο καλοκαίρι να υπάρχει μια συμφωνία που θα ακυρώνει στην πράξη το τουρκολιβυκό μνημόνιο, αφού θα το επικαλύπτει. Για την κυβέρνηση βασικό σημείο θα είναι να επικοινωνήσει εντός της χώρας για ποιον λόγο προχώρησε μια τέτοια συμφωνία με την Αίγυπτο σε αυτή τη στιγμή, που προφανώς δεν θα βασίζεται στη μέση γραμμή λόγω των ιδιαίτερων γεωγραφικών χαρακτηριστικών (νησιά έναντι αφρικανικής ενδοχώρας)."
#write_to_bin(article)

