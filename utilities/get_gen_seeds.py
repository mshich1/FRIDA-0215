import json
import random

# this script goes through the original data created in the DMR work that was subsequently multiple-choice-ified
# it then splits the data into data used for synthetic data generation and leftovers that can be used for 1-shots
# and evaluation. 
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
                # if there's left over lines, remove selected lines from current
                if len(curr_lines) > NUM_SHOT:
                    print(f"Curr lines before removal: {curr_lines}")
                    for foo in gen_lines:
                        curr_lines.remove(foo)
                    print(f"Curr lines after removal: {curr_lines}")
                # format and write out lines used for data generation
                new_gen_idx = 0
                for q in gen_lines:
                    q["name"] = f"{curr_template}_{new_gen_idx}"
                    new_gen_idx += 1
                    q["id"] = f"gen_task_{counter_gen}"
                    counter_gen += 1
                    json.dump(q, output_gen)
                    output_gen.write("\n")
                # format and write out leftover lines
                new_eval_idx = 0
                for p in curr_lines:
                    p["name"] = f"{curr_template}_{new_eval_idx}"
                    new_eval_idx += 1
                    p["id"] = f"eval_task_{counter_eval}"
                    counter_eval += 1
                    json.dump(p, output_eval)
                    output_eval.write("\n")
                # update parameters for next round
                curr_lines = [l]
                curr_template = curr_name