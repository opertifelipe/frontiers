from sklearn.model_selection import train_test_split
from train.keywords_approach import create_embeddings_keywords
from train.document_approach import create_embeddings_document

from utils.utils import IO

def train_test(df_subset):
    X_train, X_test, y_train, y_test = train_test_split(df_subset[["id","text"]],df_subset["journal"].values, 
                                                        test_size=0.33, 
                                                        random_state=42,
                                                        stratify=df_subset["journal"].values)
    df_train = X_train.copy()
    df_train["journal"] = y_train
    df_test = X_test.copy()
    df_test["journal"] = y_test
    return df_train.reset_index(drop=True), df_test.reset_index(drop=True)

def create_journal_emebeddings(df):
    journal_embeddings = df.groupby(["journal"])["embeddings"].mean().to_frame().reset_index()
    return journal_embeddings

def train_embeddings_keyword_word2vec(df_train):
    df_train = create_embeddings_keywords(df_train, "word2vec")
    journal_embeddings = create_journal_emebeddings(df_train)
    IO(journal_embeddings, "journals_embeddings_keywords_word2vec","04_model","pickle").save()

def train_embeddings_keyword_tfidf(df_train):
    df_train = create_embeddings_keywords(df_train, "tfidf")
    journal_embeddings = create_journal_emebeddings(df_train)
    IO(journal_embeddings, "journals_embeddings_keywords_tfidf","04_model","pickle").save()

def train_embeddings_document_word2vec(df_train):
    df_train = create_embeddings_document(df_train, "word2vec")
    journal_embeddings = create_journal_emebeddings(df_train)
    IO(journal_embeddings, "journals_embeddings_document_word2vec","04_model","pickle").save()
