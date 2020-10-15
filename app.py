# Front end interface for fake news classification
# Author: Vinh Nguyen
# User inputs a url for an article and app returns label 'fake' or 'real'

import streamlit as st
import pandas as pd
import numpy as np
from simpletransformers.classification import ClassificationModel
from PIL import Image

from src.utils import *

article = None
model = None

st.title('Real-time Fake News Classifier')
st.text('Created by Vinh Nguyen')
header_img = Image.open('media/header-image.jpg')
st.image(header_img, use_column_width=True)

'''
### Fake News or Real News?
Put the link of that news article you found from your local social media news feed
to curb your anxiety and find out if your go-to news station feeds you dirty lies.
'''

'''
The language model powering this app is based on **Bidirectional Encoder Representations
from Transformers (BERT)**. It learns the contextual relations between words (or sub-words)
in a text.
'''

url = st.text_input('URL of News Article')

left_button, right_button = st.beta_columns(2)

button1 = left_button.button('    Process    ')
button2 = right_button.button('Fake or Real?')

if button1:
    if url:
        with st.spinner('Reading your article...'):
            article = process_web_article(url)

            article.download()
            article.parse()

            col1, col2 = st.beta_columns(2)

            # author
            col1.subheader('Author(s)')
            col1.write(article.authors[0])

            # publish date
            col2.subheader('Date of Publish')
            col2.write(article.publish_date)

            # keywords
            col1.subheader('Keywords')
            col1.write(article.nlp())

            # summary
            st.subheader('Article Summary')
            st.write(article.summary)

            # text
            st.subheader('Full Article')
            st.write(article.text)

    else:
        st.error('Please enter a url :)')

if button2:
    if url:
        with st.spinner('Analyzing your article...'):
            article = process_web_article(url)

            model = ClassificationModel('bert', 'bert_model', use_cuda=False, from_tf=True)
            prediction = model.predict(article.text)
            st.write(prediction)
    else:
        st.error('What am I supposed to do with no url!! :(')