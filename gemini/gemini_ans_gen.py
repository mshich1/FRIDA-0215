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
    system_instruction="""You will be creating multiple choice questions on a variety of topics related to
    common sense and/or earthquake knowledge. Be creative in choosing the vocabulary and phrasing of these questions.""")

@sleep_and_retry
@limits(calls=14,period=MINUTE)
def check_lim():
    return
