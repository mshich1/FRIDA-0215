import transformers
import torch
import json

model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

file_in = input("Put in the relative directory and jsonl file name that you want to send to llama3: ")
file_out = input("Write the name of the file you want the results written to: ")

pipeline = transformers.pipeline(
    "text-generation", model=model_id, model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto"
)

print(pipeline("Hey how are you doing today?"))