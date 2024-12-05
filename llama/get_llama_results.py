import transformers
import torch
import json
from tqdm import tqdm

model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

file_in = "../seed_data/seed_tasks_eval.jsonl"
file_out = "../llama_results/llama.txt"

pipe = transformers.pipeline("text-generation", model=model_id, model_kwargs={"torch_dtype": torch.bfloat16}, max_new_tokens=200)

instruct = [json.loads(l) for l in open(file_in,"r")]
with(open(file_out, "w")) as outie:
    for i in tqdm(instruct):
        task = i["instruction"]
        choices = i["instances"][0]["input"]
        instruction = task + "\n" + choices + "\n" + "Answer: "
        res = pipe(instruction)[0]["generated_text"]
        res = ' '.join(res.split("\n"))
        outie.write(f"{res}\n")
        