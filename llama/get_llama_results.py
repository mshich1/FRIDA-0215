import transformers
import torch
import json
import re
from tqdm import tqdm

model_id = ["meta-llama/Llama-3.2-1B-Instruct","meta-llama/Llama-3.1-8B-Instruct"]

file_in = "../seed_data/seed_tasks_eval.jsonl"
file_out = ["../llama_results/llama_sm.txt","../llama_results/llama.txt"]

instruct = [json.loads(l) for l in open(file_in,"r")]
for f, m in zip(file_out,model_id):
    pipe = transformers.pipeline("text-generation", model=m, model_kwargs={"torch_dtype": torch.bfloat16}, max_new_tokens=200, device = 0)
    with(open(f, "w")) as outie:
        for i in tqdm(instruct):
            task = i["instruction"]
            choices = i["instances"][0]["input"]
            query = [{"role":"system", "content": "The following is a multiple choice question about object properties and earthquakes. There is only one correct answer. Your answer should repeat the completely correct answer exactly word for word, with no explanation."},{"role": "user", "content": f"{i['instruction']} {i['instances'][0]['input']}"}]
            res = pipe(query)[0]["generated_text"][2]["content"]
            print(res)
            to_go = ' '.join(res.split("\n"))
            outie.write(f"{to_go}\n")
        
