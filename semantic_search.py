# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 08:34:51 2024

@author: qiusi
"""

from sentence_transformers import SentenceTransformer, util
import pandas as pd
model_path = r'C:\Users\qiusi\OneDrive\Documents\8.GATECH\07_ML1\project\all-mpnet-base-v2'

def semantic_search(query, sentences):
    # 1 import model
    ## method 1: connect to hugging face
    # model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    
    ## method 2: connect to local copy
    model = SentenceTransformer(model_path)

    # 2 Encode the documents into embeddings
    print("Encode the documents into embeddings...")
    document_embeddings = model.encode(sentences, convert_to_tensor=True)

    # 3 Encode the query into an embedding
    print("Encode the query into embeddings...")
    query_embedding = model.encode(query, convert_to_tensor=True)

    # Perform semantic search
    # Find the documents with the highest cosine similarity to the query
    cosine_scores = util.pytorch_cos_sim(query_embedding, document_embeddings)[0]
    cosine_scores = cosine_scores.tolist()
    
    # save into a dataframe
    df = pd.DataFrame([sentences, cosine_scores]).T
    df.columns = ['sentences', 'score']
    df.sort_values(by='score', ascending=False, inplace=True)

    return df
            

# query = "What controls should be in place?"