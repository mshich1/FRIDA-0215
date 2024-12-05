# This is standard code for getting answers to prompts from gemini from jsonl files
# You gotta type in the name of the file

import google.generativeai as genai
import os
import json
from tqdm import tqdm
from ratelimit import limits, sleep_and_retry

MINUTE = 60

file_in = input("Put in the relative directory and jsonl file name that you want to send to gemini: ")
file_out = input("Write the name of the file you want the results written to: ")

instruct = [json.loads(l) for l in open(file_in,"r")]
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    system_instruction="Do not provide any explanations for your answer. Only choose correct answers. \
        There is only 1 correct answer. Write both the letter chosen and the associated answer. \
        For example, if the question was \"True or False, clowns have red noses. A) True\tB) False\" respond \"A) True\". \
        All answers should be given in one line")

@sleep_and_retry
@limits(calls=14,period=MINUTE)
def check_lim():
    return

gemini_ans = []
with open(file_out, "w") as results:
    for i in tqdm(instruct):
        task = i["instruction"]
        choices = i["instances"][0]["input"]
        instruction = task + "\n" + choices + "\n" + "Answer: "
        check_lim() 
        res = model.generate_content(instruction).text
        results.write(f"{res}\n")