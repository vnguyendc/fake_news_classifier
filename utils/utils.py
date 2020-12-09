# Utility Functions
import streamlit as st
import pandas as pd
import numpy as np
from newspaper import Article
from bert import run_classifier, tokenization
import tensorflow as tf
import tensorflow_hub as hub

def process_web_article(url):
    article = Article(url)
    return article

label_list = [0, 1]
MAX_SEQ_LENGTH = 128

# This is a path to an uncased (all lowercase) version of BERT
BERT_MODEL_HUB = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"

def create_tokenizer_from_hub_module():
  """Get the vocab file and casing info from the Hub module."""
  with tf.Graph().as_default():
    bert_module = hub.Module(BERT_MODEL_HUB)
    tokenization_info = bert_module(signature="tokenization_info", as_dict=True)
    with tf.Session() as sess:
      vocab_file, do_lower_case = sess.run([tokenization_info["vocab_file"],
                                            tokenization_info["do_lower_case"]])

  return tokenization.FullTokenizer(
      vocab_file=vocab_file, do_lower_case=do_lower_case)

def predict(sentences, predict_fn):
    tokenizer = create_tokenizer_from_hub_module()
    labels = [0, 1]
    input_examples = [
        run_classifier.InputExample(
            guid="",
            text_a = x,
            text_b = None,
            label = 0
        ) for x in sentences] # here, "" is just a dummy label
    input_features = run_classifier.convert_examples_to_features(
        input_examples, labels, MAX_SEQ_LENGTH, tokenizer
    )

    all_input_ids = []
    all_input_mask = []
    all_segment_ids = []
    all_label_ids = []

    for feature in input_features:
        all_input_ids.append(feature.input_ids)
        all_input_mask.append(feature.input_mask)
        all_segment_ids.append(feature.segment_ids)
        all_label_ids.append(feature.label_id)
    pred_dict = {
        'input_ids': all_input_ids,
        'input_mask': all_input_mask,
        'segment_ids': all_segment_ids,
        'label_ids': all_label_ids
    }

    predictions = predict_fn(pred_dict)
    print(predictions)
    return [
        (sentence, prediction, label)
        for sentence, prediction, label in zip(sentences, predictions['probabilities'], [predictions['labels']])
    ]