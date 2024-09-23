import json
import random

NUM_SHOT = 5
instruct = [json.loads(l) for l in open("../seed_data/seed_tasks_earthquake_mc_all.jsonl")]

with open("../seed_data/seed_tasks_earthquake_gen.jsonl","w") as output_gen:
    with open("../seed_data/seed_tasks_eval.jsonl","w") as output_eval:
        curr_template = "biggest"
        counter_eval = 0
        counter_gen = 0
        curr_lines = []
        for l in instruct:
            temp_name_arr = l["name"].split("_")
            temp_name_arr.pop()
            curr_name = "_".join(temp_name_arr)
            if curr_name==curr_template and l != instruct[-1]:
                curr_lines.append(l)
            else:
                #choose the 5 shot we'll use and save the rest for the eval set
                gen_lines = []
                if len(curr_lines) > NUM_SHOT:
                    gen_lines = random.sample(curr_lines, NUM_SHOT)
                else:
                    gen_lines = curr_lines
                    curr_lines = []
                if len(curr_lines) != NUM_SHOT:
                    for foo in gen_lines:
                        curr_lines.remove(foo)
                new_gen_idx = 0
                for q in gen_lines:
                    q["name"] = f"{curr_template}_{new_gen_idx}"
                    new_gen_idx += 1
                    q["id"] = f"gen_task_{counter_gen}"
                    counter_gen += 1
                    json.dump(q, output_gen)
                    output_gen.write("\n")
                new_eval_idx = 0
                for p in curr_lines:
                    p["name"] = f"{curr_template}_{new_eval_idx}"
                    new_eval_idx += 1
                    q["id"] = f"eval_task_{counter_gen}"
                    counter_eval += 1
                    json.dump(q, output_eval)
                    output_eval.write("\n")