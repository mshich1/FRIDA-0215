import json
import random

# This is a script I used to extract instructions from the og mc dataset to be 1-shot tests 
# of the gemini, llama3 vanilla and the llama3 instruct models. 
# The split files are earthquake_oneshots and seed_tasks_earthquake_mc_oneshot
instruct = [json.loads(l) for l in open("../seed_tasks_earthquake_mc.jsonl")]

with open("../earthquake_oneshots.jsonl","w") as output_1s:
    with open("../seed_tasks_earthquake_mc_oneshot.jsonl","w") as output_seeds:
        curr_template = "biggest"
        counter_1s = 0
        counter_seedlines = 0
        curr_template_lines = []
        for l in instruct:
            # get template that we're working with
            temp_name_arr = l["name"].split("_")
            temp_name_arr.pop()
            curr_name = "_".join(temp_name_arr)
            # collect all seeds in template
            if curr_name==curr_template and l != instruct[-1]:
                curr_template_lines.append(l)
            # get the 1 shot example
            else:
                # choose 1 shot line and reconfigure it for new file
                one_shot_out = random.choice(curr_template_lines)
                curr_template_lines.remove(one_shot_out)
                one_shot_out["name"] = curr_template
                one_shot_out["id"] = f"one_shot_{counter_1s}"
                # rewrite seed and oneshot file to reflect change
                new_idx = 0
                for bit in curr_template_lines:
                    bit["name"] = f"{curr_template}_{new_idx}"
                    new_idx += 1
                    bit["id"] = f"frida_task_{counter_seedlines}"
                    counter_seedlines += 1
                    json.dump(bit, output_seeds)
                    output_seeds.write("\n")
                json.dump(one_shot_out, output_1s)
                output_1s.write("\n")
                # reset data structures for next category
                curr_template_lines = [l]
                curr_template = curr_name
                counter_1s += 1