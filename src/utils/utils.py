import spacy
import json
import pandas as pd
import pickle

nlp = spacy.load("en_core_web_lg")

def load_data():
    with open('data/01_raw/Jan2020Frontiers.jsonl', 'r') as f:
        list_of_papers = [json.loads(line) for line in f]
    df = pd.DataFrame.from_dict(list_of_papers)
    return df

class IO:

    def __init__(self, object_ = None, filename = None, folder = None, format_ = None):
        self.object_ = object_
        self.filename = filename
        self.folder = folder
        self.format_ = format_

    def _savepickle(self):
        try:
            pickle.dump(self.object_, open(f"data/{self.folder}/{self.filename}.{self.format_}","wb"))
        except:
            pickle.dump(self.object_, open(f"data/{self.folder}/{self.filename}.{self.format_}","w"))            

    def _savejson(self):
        try:
            json.dump(self.object_, open(f"data/{self.folder}/{self.filename}.{self.format_}","wb"))
        except:
            json.dump(self.object_, open(f"data/{self.folder}/{self.filename}.{self.format_}","w"))

    def _loadpickle(self):
        try:
            self.object_ = pickle.load(open(f"data/{self.folder}/{self.filename}.{self.format_}","rb"))
        except:
            self.object_ = pickle.load(open(f"data/{self.folder}/{self.filename}.{self.format_}","r")) 

    def _loadjson(self):
        try:
            self.object_ = json.load(open(f"data/{self.folder}/{self.filename}.{self.format_}","rb"))
        except:
            self.object_ = json.load(open(f"data/{self.folder}/{self.filename}.{self.format_}","r"))

    def save(self):
        if self.format_ == "json":
            self._savejson()
        else:
            self._savepickle()     

    def load(self):
        if self.format_ == "json":
            self._loadjson()
        else:
            self._loadpickle()
        return self.object_            
