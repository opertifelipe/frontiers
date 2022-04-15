import sys
from train.train import train, train_test
from preprocess.preprocess import filter_papers_min_sample, preprocess
from utils.utils import load_data

def main(argv):
    ## Setup
    if len(argv)  == 1:
        env = argv[0]
    else:
        env = "cpu"
    assert env in ["cpu","gpu"]

    df = load_data()
    _, df = filter_papers_min_sample(df)
    df_train, df_test = train_test(df)

    ## Preprocessing
    df_train = preprocess(df_train)
    df_test = preprocess(df_test)    





if __name__ == "__main__":
    main(sys.argv[1:])