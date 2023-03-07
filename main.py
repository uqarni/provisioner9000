import openai
import pandas as pd
import os


#openai.api_key = os.environ("OPENAI_API_KEY")

#take original csv, make a duplicate, add correct jsonl, save as duplicate jsonl file
ingest = pd.read_csv('lilearning_t1.csv')

for index, row in ingest.iterrows():
    print(ingest.loc[index,"unfinished prompt"])




#split into training, validation, and test sets
#upload training and validation sets to openai
#fine-tune model
#query model
