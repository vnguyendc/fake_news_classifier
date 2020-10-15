# Utility Functions

import pandas as pd
import numpy as np
from simpletransformers.classification import ClassificationModel
from newspaper import Article

def process_web_article(url):
    article = Article(url)
    return article

def load_model():
    model = ClassificationModel('bert', 'bert_model')

    return model


def predict(model, text):
    prediction = model.predict(text)

    return prediction
