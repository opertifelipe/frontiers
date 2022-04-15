import sys
from train.train import (train_test,
                        train_embeddings_keyword_word2vec,
                        train_embeddings_document_word2vec,
                        train_embeddings_keyword_tfidf,
                        train_embeddings_document_tfidf,
                        train_embeddings_keyword_sbert,
                        train_embeddings_document_sbert
                        )                        
from preprocess.preprocess import filter_papers_min_sample, preprocess
from evaluate.evaluate import (evaluate_document_word2vec, 
                               evaluate_keyword_word2vec,
                               evaluate_keyword_tfidf,
                               evaluate_document_tfidf,
                               evaluate_keyword_sbert,
                               evaluate_document_sbert)

from utils.utils import load_data, IO
from pandarallel import pandarallel

import warnings
warnings.filterwarnings("ignore")

pandarallel.initialize(progress_bar=True)

def main(argv):
    ## Setup
    if len(argv)  == 1:
        env = argv[0]
    else:
        env = "cpu"
    assert env in ["cpu","gpu"]

    # df = load_data()
    
    # _, df = filter_papers_min_sample(df)
    # df_train, df_test = train_test(df)

    # ## Preprocessing

    # df_train = preprocess(df_train)
    # df_test = preprocess(df_test)    
    # IO(df_train, "df_train_preprocessed","02_intermediate","pickle").save()
    # IO(df_test, "df_test_preprocessed","02_intermediate","pickle").save()    

    df_train = IO(filename="df_train_preprocessed",folder="02_intermediate",format_="pickle").load()
    df_test = IO(filename="df_test_preprocessed",folder="02_intermediate",format_="pickle").load()    

    ## Training keywords
    # train_embeddings_keyword_word2vec(df_train)
    # train_embeddings_keyword_tfidf(df_train)
    # if env == "gpu":
    #     train_embeddings_keyword_sbert(df_train)

    ## Training document
    # train_embeddings_document_word2vec(df_train)
    # train_embeddings_document_tfidf(df_train)
    # if env == "gpu":
    #     train_embeddings_document_sbert(df_train)

    ## Evaluation Baseline
    # generate_baseline_evaluation(df_test)

    ### Evaluate keyword 
    # evaluate_keyword_word2vec(df_test)
    # evaluate_keyword_tfidf(df_test)
    # if env == "gpu":
    #     evaluate_keyword_sbert(df_test)

    ## Evaluate document
    # evaluate_document_word2vec(df_test)
    # evaluate_document_tfidf(df_test)
    if env == "gpu":
        evaluate_document_sbert(df_test)



if __name__ == "__main__":
    main(sys.argv[1:])