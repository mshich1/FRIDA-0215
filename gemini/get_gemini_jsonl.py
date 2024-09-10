# This is standard code for getting answers to prompts from gemini from jsonl files
# You gotta type in the name of the file

import google.generativeai as genai
import os
import json
from ratelimit import limits, sleep_and_retry

MINUTE = 60

file_in = input("Put in the relative directory and jsonl file name that you want to send to gemini: ")
file_out = input("Write the name of the file you want the results written to: ")

instruct = [json.loads(l) for l in open(file_in,"r")]
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    system_instruction="Do not provide any explanations for your answer choices. Choose all correct answers.")

@sleep_and_retry
@limits(calls=15,period=MINUTE)    
def get_ans(): 
    gemini_ans = []
    for i in instruct:
        task = i["instruction"]
        choices = i["instances"][0]["input"]
        instruction = task + "\n" + choices 
        gemini_ans.append(model.generate_content(instruction))
    return gemini_ans

ans = get_ans()
with open(file_out, "w") as results:
    for i in ans:
        results.write(f"{i}\n")    