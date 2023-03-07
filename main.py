import openai
import pandas as pd
import os


openai.api_key = os.environ.get("OPENAI_API_KEY")

#upload training and validation sets to openai
training = openai.File.create(
  file=open("training.jsonl", "rb"),
  purpose='fine-tune'
)

print(training)
#fine-tune model
#query model
