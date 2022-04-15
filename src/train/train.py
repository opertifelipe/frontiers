from sklearn.model_selection import train_test_split

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



def train_keyword_embeddings_word2vec(df_train):
    return     