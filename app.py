# Front end interface for fake news classification
# Author: Vinh Nguyen
# User inputs a url for an article and app returns label 'fake' or 'real'

import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from utils.utils import *
from tensorflow.contrib import predictor
import tensorflow as tf
from streamlit import components
import base64

import os
os.environ['GOOGLE_APPLICATION_CREDENTIALS']="fake_news.json"

article = None
model = None

TITLE = 'real-time fake news classifier :newspaper:'

st.set_page_config(page_title=TITLE, page_icon=':newspaper:')
st.title(TITLE)



header_img = Image.open('media/header-image.jpg')
st.image(header_img, use_column_width=True)

with st.beta_expander('about'):
    '''
    created by vinh nguyen\n\n
    [github](https://github.com/vnguyendc/fake_news_classifier)
    [linkedin](https://www.linkedin.com/in/vinh-nguyen-397572b9/)
    '''
    '''
    ### *fake news or real news?*
    put the link of that news article you found from your local social media news feed
    to curb your anxiety and find out if your go-to news station feeds you dirty lies.
    '''

    '''
    the language model powering this app is based on **bidirectional encoder representations
    from transformers (BERT)**. It learns the contextual relations between words (or sub-words)
    in a text.
    '''

# user input of URL of news article
url = st.text_input('Enter a URL of a news article')

left_button, right_button = st.beta_columns(2)

button1 = left_button.button('    Read    ')
button2 = right_button.button('Fake or Real?')

if button1:
    if url:
        with st.spinner('Reading your article...'):

            # parse web article
            article = process_web_article(url)
            article.download()
            article.parse()

            st.header(article.title)

            col1, col2 = st.beta_columns(2)

            # author
            col1.subheader('Author(s)')
            if article.authors:
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
            print('article processed')

            export_dir = './model/'
            predict_fn = predictor.from_saved_model(export_dir)
            print('model loaded')
            prediction = predict([article.text], predict_fn)
            print('inference successful')
            print(prediction)
            label = prediction[0][2]

            if label == 0:
                st.write("Real!!")
                components.v1.html('<div style="width:100%;height:0;padding-bottom:56%;position:relative;"><iframe src="https://giphy.com/embed/l3q2CIDIi2cnUZ2hy" width="100%" height="100%" style="position:absolute" frameBorder="0" class="giphy-embed" allowFullScreen></iframe></div><p><a href="https://giphy.com/gifs/snl-saturday-night-live-season-42-l3q2CIDIi2cnUZ2hy">via GIPHY</a></p>')
            else:
                st.write("FAKE NEWS!!")
                components.v1.html('<div style="width:100%;height:0;padding-bottom:56%;position:relative;"><iframe src="https://giphy.com/embed/l0Iyau7QcKtKUYIda" width="100%" height="100%" style="position:absolute" frameBorder="0" class="giphy-embed" allowFullScreen></iframe></div><p><a href="https://giphy.com/gifs/reactionseditor-reaction-l0Iyau7QcKtKUYIda">via GIPHY</a></p>')
    else:
        st.error('What am I supposed to do with no url!! :(')