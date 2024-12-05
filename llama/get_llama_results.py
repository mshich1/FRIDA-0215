import transformers
import torch
import json
import re
from tqdm import tqdm

model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

file_in = "../seed_data/seed_tasks_eval.jsonl"
file_out = "../llama_results/llama.txt"

pipe = transformers.pipeline("text-generation", model=model_id, model_kwargs={"torch_dtype": torch.bfloat16}, max_new_tokens=200, device = 0)

instruct = [json.loads(l) for l in open(file_in,"r")]
with(open(file_out, "w")) as outie:
    for i in tqdm(instruct):
        task = i["instruction"]
        choices = i["instances"][0]["input"]
        query = [{"role":"system", "content": "The following is a multiple choice question about object properties and earthquakes. There is only one correct answer. Your answer should repeat the correct answer exactly with no explanation."},{"role": "user", "content": f"{i['instruction']} {i['instances'][0]['input']}"}]
        res = pipe(query)[0]["generated_text"]
        ans_matched = re.search("<\|start_header_id\|>assistant<\|end_header_id\|>([\s\S]*)<\|eot_id\|>", res)
        if ans_matched is None:
            ans_matched = re.search("<\|start_header_id\|>assistant<\|end_header_id\|>([\s\S]*)", res)
        to_go = ans_matched.group(1)
        to_go = ' '.join(to_go.split("\n"))
        outie.write(f"{to_go}\n")
        