import transformers
import torch
import json

model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

file_in = input("Put in the relative directory and jsonl file name that you want to send to llama3: ")
file_out = input("Write the name of the file you want the results written to: ")

pipe = transformers.pipeline("text-generation", model=model_id, model_kwargs={"torch_dtype": torch.bfloat16}).to("cuda")

instruct = [json.loads(l) for l in open(file_in,"r")]
with(open(file_out, "w")) as outie:
    for i in instruct:
        task = i["instruction"]
        choices = i["instances"][0]["input"]
        instruction = task + "\n" + choices + "\n" + "Answer: "
        res = pipe(instruction)[0]["generated_text"]
        res = ' '.join(res.split("\n"))
        outie.write(f"{res}\n")
        