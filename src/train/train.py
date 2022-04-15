

def create_journal_emebeddings(df):
    journal_embeddings = df.groupby(["journal"])["embeddings"].mean().to_frame().reset_index()
    return journal_embeddings



def train_keyword_embeddings_word2vec(df_train):
    return     