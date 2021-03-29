# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
# Modifications Copyright 2017 Abigail See
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""This file contains code to run beam search decoding, including running ROUGE evaluation and producing JSON datafiles for the in-browser attention visualizer, which can be found here https://github.com/abisee/attn_vis"""

import os
import time
import tensorflow as tf
import beam_search
import data
import json
import pyrouge
import util
import logging
import numpy as np
from rouge import Rouge

FLAGS = tf.app.flags.FLAGS

SECS_UNTIL_NEW_CKPT = 60  # max number of seconds before loading new checkpoint

article_withunks_list = []
abstract_withunks_list=[]
decoded_output_list = []

class BeamSearchDecoder(object):
  """Beam search decoder."""

  def __init__(self, model, batcher, vocab):
    """Initialize decoder.

    Args:
      model: a Seq2SeqAttentionModel object.
      batcher: a Batcher object.
      vocab: Vocabulary object
    """
    self._model = model
    self._model.build_graph()
    self._batcher = batcher
    self._vocab = vocab
    self._saver = tf.train.Saver() # we use this to load checkpoints for decoding
    self._sess = tf.Session(config=util.get_config())

    # Load an initial checkpoint to use for decoding
    ckpt_path = util.load_ckpt(self._saver, self._sess)

    if FLAGS.single_pass:
      # Make a descriptive decode directory name
      ckpt_name = "ckpt-" + ckpt_path.split('-')[-1] # this is something of the form "ckpt-123456"
      self._decode_dir = os.path.join(FLAGS.log_root, get_decode_dir_name(ckpt_name))
      if os.path.exists(self._decode_dir):
        raise Exception("single_pass decode directory %s should not already exist" % self._decode_dir)

    else: # Generic decode dir name
      self._decode_dir = os.path.join(FLAGS.log_root, "decode")

    # Make the decode dir if necessary
    if not os.path.exists(self._decode_dir): os.mkdir(self._decode_dir)

    if FLAGS.single_pass:
      # Make the dirs to contain output written in the correct format for pyrouge
      self._rouge_ref_dir = os.path.join(self._decode_dir, "reference")
      if not os.path.exists(self._rouge_ref_dir): os.mkdir(self._rouge_ref_dir)
      self._rouge_dec_dir = os.path.join(self._decode_dir, "decoded")
      if not os.path.exists(self._rouge_dec_dir): os.mkdir(self._rouge_dec_dir)

  def decode_inf(self):
    batch = self._batcher.next_batch()  # 1 example repeated across batch

    original_article = batch.original_articles[0]  # string
    original_abstract = batch.original_abstracts[0]  # string

    # input data
    article_withunks = data.show_art_oovs(original_article, self._vocab) # string
    abstract_withunks = data.show_abs_oovs(original_abstract, self._vocab, (batch.art_oovs[0] if FLAGS.pointer_gen else None)) # string

    # Run beam search to get best Hypothesis
    best_hyp = beam_search.run_beam_search(self._sess, self._model, self._vocab, batch)

    # Extract the output ids from the hypothesis and convert back to words
    output_ids = [int(t) for t in best_hyp.tokens[1:]]
    decoded_words = data.outputids2words(output_ids, self._vocab, (batch.art_oovs[0] if FLAGS.pointer_gen else None))

    # Remove the [STOP] token from decoded_words, if necessary
    try:
      fst_stop_idx = decoded_words.index(data.STOP_DECODING) # index of the (first) [STOP] symbol
      decoded_words = decoded_words[:fst_stop_idx]
    except ValueError:
      decoded_words = decoded_words
    decoded_output = ' '.join(decoded_words) # single string

    #sys.stdout.write(decoded_output)
    output_sum = 'Summary_' + time.strftime("%Y%m%d-%H%M%S") + '.txt'

    with open(output_sum, 'w') as writer:
      writer.write(decoded_output)
    print("Summary is : ")
    print(decoded_output)
    


  def decode(self):
    """Decode examples until data is exhausted (if FLAGS.single_pass) and return, or decode indefinitely, loading latest checkpoint at regular intervals"""
    t0 = time.time()
    counter = 0
    countval=0
    countsame=0
    while True:
      batch = self._batcher.next_batch()  # 1 example repeated across batch
      if batch is None: # finished decoding dataset in single_pass mode
        assert FLAGS.single_pass, "Dataset exhausted, but we are not in single_pass mode"
        tf.logging.info("Decoder has finished reading dataset for single_pass.")
        tf.logging.info("Output has been saved in %s and %s. Now starting ROUGE eval...", self._rouge_ref_dir, self._rouge_dec_dir)
        results_dict = rouge_eval(self._rouge_ref_dir, self._rouge_dec_dir)
        rouge_log(results_dict, self._decode_dir)
        return

      original_article = batch.original_articles[0]  # string
      original_abstract = batch.original_abstracts[0]  # string
      original_abstract_sents = batch.original_abstracts_sents[0]  # list of strings

      article_withunks = data.show_art_oovs(original_article, self._vocab) # string
      abstract_withunks = data.show_abs_oovs(original_abstract, self._vocab, (batch.art_oovs[0] if FLAGS.pointer_gen else None)) # string

      # Run beam search to get best Hypothesis
      best_hyp = beam_search.run_beam_search(self._sess, self._model, self._vocab, batch)

      # Extract the output ids from the hypothesis and convert back to words
      output_ids = [int(t) for t in best_hyp.tokens[1:]]
      decoded_words = data.outputids2words(output_ids, self._vocab, (batch.art_oovs[0] if FLAGS.pointer_gen else None))

      # Remove the [STOP] token from decoded_words, if necessary
      try:
        fst_stop_idx = decoded_words.index(data.STOP_DECODING) # index of the (first) [STOP] symbol
        decoded_words = decoded_words[:fst_stop_idx]
      except ValueError:
        decoded_words = decoded_words
      decoded_output = ' '.join(decoded_words) # single string
      

      if FLAGS.single_pass:
        counter += 1
        #print(len(abstract_withunks.split()))
        #print(len(decoded_output.split()))
        if (len(abstract_withunks.split()) <= len(decoded_output.split())):
          countsame=findstr(abstract_withunks,decoded_output)
          #countsame=findstr(decoded_output,abstract_withunks)
        elif (len(abstract_withunks.split()) > len(decoded_output.split())):
          countsame=findstr(decoded_output,abstract_withunks)
          #countsame=findstr(abstract_withunks,decoded_output)
        if (countsame < 3) :
          countval += 1 # this is how many examples we've decoded
          self.write_for_rouge(original_abstract_sents, decoded_words, counter) # write ref summary and decoded summary to file, to eval with pyrouge later
          # counter += 1 # this is how many examples we've decoded
          rouge_results(article_withunks, abstract_withunks, decoded_output)
          article_withunks_list.append(article_withunks)
          abstract_withunks_list.append(abstract_withunks)
          decoded_output_list.append(decoded_output)
        if(counter == 1000):
          tf.logging.info("stopped at 1000 samples")
          rouge_avg(decoded_output_list, abstract_withunks_list,countval)
          print("Number of summaries taking account into average Rouge score are : ")
          print(countval)
      else:
        print_results(article_withunks, abstract_withunks, decoded_output) # log output to screen
        self.write_for_attnvis(article_withunks, abstract_withunks, decoded_words, best_hyp.attn_dists, best_hyp.p_gens) # write info to .json file for visualization tool
        rouge_results(article_withunks, abstract_withunks, decoded_output)
        article_withunks_list.append(article_withunks)
        abstract_withunks_list.append(abstract_withunks)
        decoded_output_list.append(decoded_output)
        counter += 1 # this is how many examples we've decoded
        if counter == 10: #just stop when reading first 10 samples
          tf.logging.info("stopped at 10 samples")
          #tf.logging.info("Now starting zaksum eval to %s...", self._decode_dir )
          #zaksum(article_withunks_list,abstract_withunks_list ,decoded_output_list ,self._decode_dir)
          #scores = rouge.get_scores(decoded_output_list, abstract_withunks_list, avg=True)
          rouge_avg(decoded_output_list, abstract_withunks_list,counter)
          return

        # Check if SECS_UNTIL_NEW_CKPT has elapsed; if so return so we can load a new checkpoint
        t1 = time.time()
        if t1-t0 > SECS_UNTIL_NEW_CKPT:
          tf.logging.info('We\'ve been decoding with same checkpoint for %i seconds. Time to load new checkpoint', t1-t0)
          _ = util.load_ckpt(self._saver, self._sess)
          t0 = time.time()

  def write_for_rouge(self, reference_sents, decoded_words, ex_index):
    """Write output to file in correct format for eval with pyrouge. This is called in single_pass mode.

    Args:
      reference_sents: list of strings
      decoded_words: list of strings
      ex_index: int, the index with which to label the files
    """
    # First, divide decoded output into sentences
    decoded_sents = []
    while len(decoded_words) > 0:
      try:
        fst_period_idx = decoded_words.index(".")
      except ValueError: # there is text remaining that doesn't end in "."
        fst_period_idx = len(decoded_words)
      sent = decoded_words[:fst_period_idx+1] # sentence up to and including the period
      decoded_words = decoded_words[fst_period_idx+1:] # everything else
      decoded_sents.append(' '.join(sent))

    # pyrouge calls a perl script that puts the data into HTML files.
    # Therefore we need to make our output HTML safe.
    decoded_sents = [make_html_safe(w) for w in decoded_sents]
    reference_sents = [make_html_safe(w) for w in reference_sents]

    # Write to file
    ref_file = os.path.join(self._rouge_ref_dir, "%06d_reference.txt" % ex_index)
    decoded_file = os.path.join(self._rouge_dec_dir, "%06d_decoded.txt" % ex_index)

    with open(ref_file, "w") as f:
      for idx,sent in enumerate(reference_sents):
        f.write(sent) if idx==len(reference_sents)-1 else f.write(sent+"\n")
    with open(decoded_file, "w") as f:
      for idx,sent in enumerate(decoded_sents):
        f.write(sent) if idx==len(decoded_sents)-1 else f.write(sent+"\n")

    tf.logging.info("Wrote example %i to file" % ex_index)


  def write_for_attnvis(self, article, abstract, decoded_words, attn_dists, p_gens):
    """Write some data to json file, which can be read into the in-browser attention visualizer tool:
      https://github.com/abisee/attn_vis

    Args:
      article: The original article string.
      abstract: The human (correct) abstract string.
      attn_dists: List of arrays; the attention distributions.
      decoded_words: List of strings; the words of the generated summary.
      p_gens: List of scalars; the p_gen values. If not running in pointer-generator mode, list of None.
    """
    article_lst = article.split() # list of words
    decoded_lst = decoded_words # list of decoded words
    to_write = {
        'article_lst': [make_html_safe(t) for t in article_lst],
        'decoded_lst': [make_html_safe(t) for t in decoded_lst],
        'abstract_str': make_html_safe(abstract),
        'attn_dists': attn_dists
    }
    if FLAGS.pointer_gen:
      to_write['p_gens'] = p_gens
    output_fname = os.path.join(self._decode_dir, 'attn_vis_data.json')
    with open(output_fname, 'w') as output_file:
      json.dump(to_write, output_file)
    tf.logging.info('Wrote visualization data to %s', output_fname)


def print_results(article, abstract, decoded_output):
  """Prints the article, the reference summmary and the decoded summary to screen"""
  print ("")
  tf.logging.info('ARTICLE:  %s', article)
  tf.logging.info('REFERENCE SUMMARY: %s', abstract)
  tf.logging.info('GENERATED SUMMARY: %s', decoded_output)
  print ("")

#////////Function for evaluate with Rouge metric and print th results on a txt file/////////#
def findstr(s1,s2):
    cnt=0
    w1=s1.split()
    w2=s2.split()
    for i in range(len(w1)):
        #print(i)
        #print(i)
        #print(w1[i])
        #print(w2[i])
        if w1[i] == w2[i]:
            cnt=cnt+1
    return cnt

def rouge_results(article,abstract,decoded_output):
    """Evaluates with Rouge metric and writes the results to rouge_results file"""
    rouge = Rouge()
    score = rouge.get_scores(decoded_output, abstract)
    with open('rouge_results256.txt', 'a+') as f:
        print("\n", file=f)
        print("\n", file=f)
        print("\n", file=f)
        print("ARTICLE:  ", file=f)
        print(article, file=f)
        print("REFERENCE SUMMARY:  ", file=f)
        print(abstract, file=f)   
        print("GENERATED SUMMARY:  ", file=f)
        print(decoded_output, file=f)
        print("ROUGE EVALUATION:  ", file=f)
        print(score, file=f)
    f.close()
    #return score

def rouge_avg(abstract,decoded_output,countval):
    rouge = Rouge()
    scores = rouge.get_scores(decoded_output, abstract, avg=True)
    with open('rouge_results256.txt', 'r') as f:
            lines = f.readlines() #read
            #lines[0]= REstr(countval)
            lines[0] = "Average Rouge Evaluation Score on " + str(countval) + " test samples :  \n" 
            lines[1] = str(scores)
    with open("rouge_results256.txt", "w") as f:
        f.writelines(lines) #write back
    f.close()

def make_html_safe(s):
  """Replace any angled brackets in string s to avoid interfering with HTML attention visualizer."""
  s.replace("<", "&lt;")
  s.replace(">", "&gt;")
  return s


def rouge_eval(ref_dir, dec_dir):
  """Evaluate the files in ref_dir and dec_dir with pyrouge, returning results_dict"""
  r = pyrouge.Rouge155()
  r.model_filename_pattern = '#ID#_reference.txt'
  r.system_filename_pattern = '(\d+)_decoded.txt'
  r.model_dir = ref_dir
  r.system_dir = dec_dir
  logging.getLogger('global').setLevel(logging.WARNING) # silence pyrouge logging
  rouge_results = r.convert_and_evaluate()
  return r.output_to_dict(rouge_results)


def rouge_log(results_dict, dir_to_write):
  """Log ROUGE results to screen and write to file.

  Args:
    results_dict: the dictionary returned by pyrouge
    dir_to_write: the directory where we will write the results to"""
  log_str = ""
  for x in ["1","2","l"]:
    log_str += "\nROUGE-%s:\n" % x
    for y in ["f_score", "recall", "precision"]:
      key = "rouge_%s_%s" % (x,y)
      key_cb = key + "_cb"
      key_ce = key + "_ce"
      val = results_dict[key]
      val_cb = results_dict[key_cb]
      val_ce = results_dict[key_ce]
      log_str += "%s: %.4f with confidence interval (%.4f, %.4f)\n" % (key, val, val_cb, val_ce)
  tf.logging.info(log_str) # log to screen
  results_file = os.path.join(dir_to_write, "ROUGE_results.txt")
  tf.logging.info("Writing final ROUGE results to %s...", results_file)
  with open(results_file, "w") as f:
    f.write(log_str)

def get_decode_dir_name(ckpt_name):
  """Make a descriptive name for the decode dir, including the name of the checkpoint we use to decode. This is called in single_pass mode."""

  if "train" in FLAGS.data_path: dataset = "train"
  elif "val" in FLAGS.data_path: dataset = "val"
  elif "test" in FLAGS.data_path: dataset = "test"
  else: raise ValueError("FLAGS.data_path %s should contain one of train, val or test" % (FLAGS.data_path))
  dirname = "decode_%s_%imaxenc_%ibeam_%imindec_%imaxdec" % (dataset, FLAGS.max_enc_steps, FLAGS.beam_size, FLAGS.min_dec_steps, FLAGS.max_dec_steps)
  if ckpt_name is not None:
    dirname += "_%s" % ckpt_name
  return dirname
