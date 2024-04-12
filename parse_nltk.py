# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 08:31:56 2024

@author: qiusi
"""

# import nltk
# nltk.download('punkt')
from nltk.tokenize import sent_tokenize

def parse_text_into_sentences(text):
    # Tokenize the text into sentences using NLTK
    sentences = sent_tokenize(text)
    return sentences