import google.generativeai as genai
import os
import json
import time
import random
import re
import string
from functools import partial
from multiprocessing import Pool
from ratelimit import limits, sleep_and_retry

import numpy as np
import tqdm
from rouge_score import rouge_scorer

import fire

CATEGORY = ["biggest","heaviest","fits","interact","can_do","can_do_size","can_do_shape","can_do_char","can_do_goal",\
    "difference","diff_criteria", "use_as", "is_a","types_of","injury","danger","damage_to_obj","explain_use",\
        "equip_used","equip_in_task","obj_loc","objs_in_loc","secondary_use","often_use","know_use","earthquake","instruct","followup"]
# batch_selfinstruct_generate.py

# run:
# python -m gemini_ans_gen generate_instruction_following_data --input_dir ../seed_data/seed_tasks_earthquake_gen.jsonl \
#  --output_dir ../gemini_results/ \
#  --num_instructions_to_generate 5 \
#  --mod_name="gemini-1.5-flash" 


MINUTE = 60
NUM_PROMPT_INSTRUCTIONS = 5

@sleep_and_retry
@limits(calls=14,period=MINUTE)
def check_lim():
    return

def encode_prompt(prompt_instructions, prompt):
    # Encode multiple prompt instructions into a single string.
    prompt_plus_fewshot = prompt + "\n"

    for idx, task_dict in enumerate(prompt_instructions):
        (instruction, input, output) = task_dict["instruction"], task_dict["input"], task_dict["output"]

        prompt_plus_fewshot += f"{{\"instruction\":{instruction},\"input\":{input},\"output\":{output}}}"
        prompt_plus_fewshot += "\n"
    return prompt_plus_fewshot

def post_process_response(num_prompt_instructions, response):
    if response is None:
        return []
    resp_split = response.text.split("\n")
    for r in resp_split:
        if r == '':
            resp_split.remove('')
    resp_split = resp_split[1:-1]
    raw_instructions = [json.loads(l, strict = False) for l in resp_split]
    # raw_instructions = re.split("###", raw_instructions)
    instructions = []
    for i in raw_instructions:
        # if the decoding stops due to length, the last example is likely truncated so we discard it
        # if idx == len(raw_instructions) - 1 and response["finish_reason"] == "length":
        #     continue
        # idx += num_prompt_instructions + 1
        # splitted_data = re.split(f"{idx}\.\s+(Instruction|Input|Output):", inst)
        # if len(splitted_data) != 7:
        #     continue
        # else:
        inst = i["instruction"]
        input = i["input"]
        output = i["output"]
        # filter based on keywords that are not suitable for language models.
        blacklist = [
            "image",
            "images",
            "graph",
            "graphs",
            "picture",
            "pictures",
            "file",
            "files",
            "map",
            "maps",
            "draw",
            "plot",
            "go to",
            "video",
            "audio",
            "music",
            "flowchart",
            "diagram",
        ]
        blacklist += []
        if any(find_word_in_string(word, inst) for word in blacklist):
            continue
        # We found that the model tends to add "write a program" to some existing instructions, which lead to a lot of such instructions.
        # And it's a bit comfusing whether the model need to write a program or directly output the result.
        # Here we filter them out.
        # Note this is not a comprehensive filtering for all programming instructions.
        if inst.startswith("Write a program"):
            continue
        # filter those starting with punctuation
        if inst[0] in string.punctuation:
            continue
        # filter those starting with non-english character
        if not inst[0].isascii():
            continue
        instructions.append({"instruction": inst, "input": input, "output": output})
    return instructions


def find_word_in_string(w, s):
    return re.compile(r"\b({0})\b".format(w), flags=re.IGNORECASE).search(s)


def generate_instruction_following_data(
    output_dir="../gemini_results/",
    input_dir="../seed_data/seed_tasks_earthquake_gen.jsonl",
    num_instructions_to_generate=40,
    mod_name="gemini-1.5-flash",
    temperature=1.1,
    num_cpus=1,
):
    for cat in CATEGORY:
        print(f"***CATEGORY IS {cat}***")
        # get the relevent seed instructions for each category
        instruct = [json.loads(l) for l in open(input_dir,"r")]
        seed_instructs = [{"instruction": t["instruction"], "input": t["instances"][0]["input"], "output": t["instances"][0]["output"]} for t in instruct if t["cat"] == cat]
        print(f"Loaded {len(seed_instructs)} human-written seed instructions")

        os.makedirs(output_dir, exist_ok=True)
        request_idx = 0
        # load the LM-generated instructions
        machine_instruction_data = []
        if os.path.exists(os.path.join(output_dir, f"{cat}.json")):
            machine_instruction_data = json.load(open(os.path.join(output_dir, f"{cat}.json")))
            print(f"Loaded {len(machine_instruction_data)} machine-generated instructions")

        # similarities = {}
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)

        # set up gemini chit chat
        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        model = genai.GenerativeModel(
            model_name=mod_name,
            system_instruction="""You will be creating multiple choice questions on a variety of topics related to
            common sense and/or earthquake knowledge. Be creative in choosing the vocabulary and phrasing of these questions.
            All responses must be given as json objects with the following format: \{\"instruction\":\"example instruction\", \"input\":\"A) this\tB) is\tC) an\tD) example\t E) question",\"output\":\"E) Question\"\}""")
        
        progress_bar = tqdm.tqdm(total=num_instructions_to_generate)
        if machine_instruction_data:
            progress_bar.update(len(machine_instruction_data))

        # tokenize instruction for ROUGE scoring
        all_instructions = [d["instruction"] for d in seed_instructs] + [
            d["instruction"] for d in machine_instruction_data
        ]
        all_instruction_tokens = [scorer._tokenizer.tokenize(inst) for inst in all_instructions]
        
        while len(machine_instruction_data) < num_instructions_to_generate:
            request_idx += 1
            prompt_descs = [json.loads(l) for l in open("./gemini_gen_prompts.jsonl","r")]
            prompt_desc = [p for p in prompt_descs if p["cat"] == cat][0]["prompt"]
            prompt = encode_prompt(seed_instructs, prompt_desc)

            request_start = time.time()
            check_lim()
            results = model.generate_content(prompt)
            request_duration = time.time() - request_start

            process_start = time.time()
            instruction_data = []
            try:
                new_instructions = post_process_response(NUM_PROMPT_INSTRUCTIONS, results)
            except:
                continue
            instruction_data += new_instructions

            total = len(instruction_data)
            keep = 0
            for instruction_data_entry in instruction_data:
                # computing similarity with the pre-tokenzied instructions
                instr_temp = instruction_data_entry["instruction"]
                input_temp = instruction_data_entry["input"]
                to_tokenize = f"{instr_temp} {input_temp}"
                new_instruction_tokens = scorer._tokenizer.tokenize(to_tokenize)
                with Pool(num_cpus) as p:
                    rouge_scores = p.map(
                        partial(rouge_scorer._score_lcs, new_instruction_tokens),
                        all_instruction_tokens,
                    )
                rouge_scores = [score.fmeasure for score in rouge_scores]
                max_score = max(rouge_scores)
                most_similar_instructions = {
                    all_instructions[i]: rouge_scores[i] for i in np.argsort(rouge_scores)[-10:][::-1]
                }
                if  max_score > 0.8 and cat not in ['use_as','is_a','often_use','know_use']:
                    continue
                elif max_score > 0.9 and cat in ['use_as','is_a']:
                    continue
                elif max_score > 0.97 and cat in ['often_use','know_use']:
                    continue
                else:
                    keep += 1
                instruction_data_entry["most_similar_instructions"] = most_similar_instructions
                instruction_data_entry["avg_similarity_score"] = float(np.mean(rouge_scores))
                machine_instruction_data.append(instruction_data_entry)
                all_instructions.append(instruction_data_entry["instruction"])
                all_instruction_tokens.append(new_instruction_tokens)
                progress_bar.update(1)
            process_duration = time.time() - process_start
            print(f"Request {request_idx} took {request_duration:.2f}s, processing took {process_duration:.2f}s")
            print(f"Generated {total} instructions, kept {keep} instructions")
            out = open(os.path.join(output_dir, f"{cat}.json"),"w")
            json.dump(machine_instruction_data, out)
        


def main(task, **kwargs):
    globals()[task](**kwargs)


if __name__ == "__main__":
    fire.Fire(main)
