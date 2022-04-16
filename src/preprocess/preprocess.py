from nltk import word_tokenize
from nltk.corpus import stopwords
import string
import nltk
import re
from summa import keywords


nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

def filter_papers_min_sample(df):
    documents_per_journal = df.groupby(["journal"]).size().sort_values(ascending=False)
    subset_journals = list(documents_per_journal[documents_per_journal >1 ].index)
    df_subset = df[df["journal"].isin(subset_journals)]
    return documents_per_journal, df_subset

def find_keywords_rule_based(s):
    start = "Keywords:"
    end = "Citation:"
    s = s.replace(" :",":")
    keywords_list = s[s.find(start)+len(start):s.rfind(end)].replace("\n"," ").replace(":","").split(",")
    keywords_list = [keyword.strip() for keyword in keywords_list ]
    return keywords_list

def preprocess_text(text):
    word_to_remove = ["google", "scholar"]
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    stop = set(stopwords.words('english') + list(string.punctuation))
    preprocessed_text = " ".join([word for word in word_tokenize(text.lower()) if (word not in stop) and (len(word) > 3) and (word not in word_to_remove)])
    return preprocessed_text

def clean_keywords(row):
    if (row["keywords_len"] < 3) or (row["keywords_len"] > 8): 
        TR_keywords = keywords.keywords(row["text"], scores=True)
        return [key[0] for key in TR_keywords[0:5]]
    else:
        return row["keywords"]    

def preprocess(df):
    df["preprocessed_text"] = df["text"].parallel_apply(preprocess_text)
    df["keywords"] = df["text"].parallel_apply(find_keywords_rule_based)
    df["keywords_len"] = df["keywords"].parallel_apply(lambda x: len(x))   
    df["keywords_cleaned"] = df.parallel_apply(clean_keywords, axis=1)
    df["keywords_cleaned_len"] =  df["keywords_cleaned"].parallel_apply(lambda x: len(x)) 
    return df
