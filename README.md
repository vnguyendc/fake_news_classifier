# Fake News Classifier Web App with Deep Learning Language Model BERT

Author: Vinh Nguyen

[Deploying huggingface‘s BERT to production with pytorch/serve](https://medium.com/analytics-vidhya/deploy-huggingface-s-bert-to-production-with-pytorch-serve-27b068026d18)

[Deploy it: Using Heroku to continuously build and deploy a deep-learning powered web application](https://medium.com/@0D0AResearch/deploy-it-using-heroku-to-continuously-build-and-deploy-a-deep-learning-powered-web-application-d3ebb41a74c7)


## Data

Data was acquired from [Kaggle](https://www.kaggle.com/c/fake-news/data). Author had aggregated various datasets across Kaggle
pertaining to fake or real news.

The total dataset size is 5384 articles. They were labelled as either fake or real. Our training dataset included 4434 articles, while our test dataset included 950 articles.

## Training our BERT Model

The model I utilized is BERT: Bidirectional Encoder Representations from Transformers. It’s a neural network architecture designed by Google researchers that’s totally transformed what’s state-of-the-art for NLP tasks, like text classification, translation, summarization, and question answering.

My preprocessing steps:

1. Lowercase our text (if we're using a BERT lowercase model)
2. Tokenize it (i.e. "sally says hi" -> ["sally", "says", "hi"])
3. Break words into WordPieces (i.e. "calling" -> ["call", "##ing"])
4. Map our words to indexes using a vocab file that BERT provides
5. Add special "CLS" and "SEP" tokens (see the [readme](https://github.com/google-research/bert))
6. Append "index" and "segment" tokens to each input (see the [BERT paper](https://arxiv.org/pdf/1810.04805.pdf))

For training, we had the following parameters:

* Max Sequence Length = 128
* Batch Size = 32
* Learning Rate = 2e-5
* Epochs = 12
* Warmup Proportion = 0.1

Training only took 1.429 minutes. This is the beauty of BERT's parallel structure.

### Evaluation

    auc = 0.9884099
    f1_score = 0.9885057
    loss = 0.0650449
    precision = 0.983368
    recall = 0.99369746

## Web App
![image](media/web_app_ss.png)

**To run locally:**

Install requirements:
`pip install requirements.txt`

Run Streamlit app:
`streamlit run app.py`

## Production

TODO:

Deploy to Heroku server as a web app.