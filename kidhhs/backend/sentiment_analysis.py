import torch.cuda
from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig
import numpy as np
from scipy.special import softmax


def preprocess(text):
    """
    Preprocesses the input text.
    Removes @user_name by generic @user and links by http.

    Source: https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest

    :param text:
    :return:
    """
    new_text = []
    text = str(text)
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)


MODEL = f"cardiffnlp/twitter-roberta-base-sentiment-latest"
# load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL)
config = AutoConfig.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)
# move the model to the gpu, if available
if torch.cuda.is_available():
    model = model.cuda()


def sentiment_for_text_batch(text_batch):
    """
    Returns a list of sentiment scores for the given batch of texts.
    :param text_batch:
    :return: returns a sentiment score for the given text.
             score is between 0 and 1 where 1 represents a positive sentiment.
    """
    # preprocess every text
    texts = [preprocess(text) for text in text_batch]
    # encode every text
    encoded_input = tokenizer(texts, return_tensors='pt', padding=True)

    # move the input to the gpu, if available
    if torch.cuda.is_available():
        for key, value in encoded_input.items():
            encoded_input[key] = value.cuda()

    # compute the model's output
    output = model(**encoded_input)
    # move it to the cpu
    scores = output[0].detach().cpu().numpy()
    scores = softmax(scores, axis=-1)

    # we multiply the scores with the following vector 0 for negative, 0.5 for neutral, 1 for positive.
    # Thereby, we can represent the three classes using a single sentiment score.
    scores_multiplier = np.asarray([[0, 0.5, 1.]])
    score = np.sum(scores * scores_multiplier, axis=-1)
    return score.tolist()


def sentiment_for_text(text):
    """
    Returns a single sentiment for a single text.
    :param text:
    :return:
    """
    return sentiment_for_text_batch([text])[0]

