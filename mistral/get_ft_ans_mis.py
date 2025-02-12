from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

import json
import os
import re
from tqdm import tqdm

base_model_name = "mistralai/Ministral-8B-Instruct-2410" #path/to/your/model/or/name/on/hub"
adapter_model_path = "../../"
adapter_model_names = ["rel_size","can_do_it","is_a_dif","risky","equip","obj_facts","quake","instr","all"]
adapter_suffix = "_mistral"
tokenized = AutoTokenizer.from_pretrained(base_model_name)
tokenized.pad_token = tokenized.eos_token

eval_qs = [json.loads(l) for l in open("../seed_data/seed_tasks_eval.jsonl")]
for a in adapter_model_names:
    print(f"***Currently testing Model {a}***")
    model = AutoModelForCausalLM.from_pretrained(base_model_name)
    model = PeftModel.from_pretrained(model, os.path.join(adapter_model_path, f"{a}{adapter_suffix}")).to("cuda")
    
    with open(f"../mistral_results/{a}{adapter_suffix}.txt","w") as ans_spot:
        for ev in tqdm(eval_qs):
            query = [{"role":"system", "content": "The following is a multiple choice question about object properties and earthquakes. There is only one correct answer. Your answer should repeat the correct answer exactly with no explanation."},{"role": "user", "content": f"{ev['instruction']} {ev['instances'][0]['input']}"}]
            chat = tokenized.apply_chat_template(query, tokenize=False, add_generation_prompt=True)
            tokenized_chat = tokenized.encode(chat, return_tensors="pt", padding=True).to("cuda")
            output = model.generate(tokenized_chat, max_new_tokens=128, pad_token_id=tokenized.eos_token_id)
            ans = tokenized.decode(output[0])
            print(f"Answer is \n{ans}")
            ans_matched = re.search("<\|start_header_id\|>assistant<\|end_header_id\|>([\s\S]*)<\|eot_id\|>", ans)
            if ans_matched is None:
                ans_matched = re.search("<\|start_header_id\|>assistant<\|end_header_id\|>([\s\S]*)", ans)
            to_go = ans_matched.group(1)
            to_go = ' '.join(to_go.split('\n'))
            ans_spot.write(to_go)
            ans_spot.write("\n")

