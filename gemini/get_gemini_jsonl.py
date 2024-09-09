# This is standard code for getting answers to prompts from gemini from jsonl files
# You gotta type in the name of the file

import google.generativeai as genai
import os
import json

file_name = input("put in the relative directory and jsonl file name that you want to send: ")

instruct = [json.loads(l) for l in open(file_name,"r")]
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    system_instruction="Do not provide any explanations for your answer choices.")
response = model.generate_content("Write a song about puffins.")
print(response.text)
print(response)