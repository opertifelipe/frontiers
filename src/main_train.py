import sys
from train.train import (train_test,
                        train_embeddings_keyword_word2vec,
                        train_embeddings_document_word2vec
                        )
from train.keywords_approach import create_embeddings_keywords                        
from preprocess.preprocess import filter_papers_min_sample, preprocess
from evaluate.evaluate import predict, generate_evaluation_report, generate_baseline_evaluation
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

    ## Training document
    train_embeddings_document_word2vec(df_train)
    

    ## Evaluation Baseline
    #generate_baseline_evaluation(df_test)

    ### Evaluate keyword 
    ## Word2Vec
    # journal_embeddings = IO(filename="journals_embeddings_keywords_word2vec",folder="04_model",format_="pickle").load()
    # df_evaluation = predict(df=df_test, 
    #                         embeddings_function=create_embeddings_keywords,
    #                         embedding_type="word2vec",
    #                         journal_embeddings=journal_embeddings)
    
    # evaluation = generate_evaluation_report(df_evaluation["journal"].tolist(),
    #                                        df_evaluation["prediction"].tolist())
    # IO(evaluation, filename="evaluation_keywords_word2vec",folder="05_report",format_="json").save()




if __name__ == "__main__":
    main(sys.argv[1:])