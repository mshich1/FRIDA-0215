import json

inputs = [json.loads(l) for l in open("../seed_data/seed_tasks_eval2.jsonl")]
with open("../seed_data/seed_tasks_eval.jsonl", "w") as outer:
    counter = 0
    new_vals = []
    for i in inputs:
        temp_name_arr = i["name"].split("_")
        temp_name_arr.pop()
        curr_name = "_".join(temp_name_arr)
        i["name"] = curr_name
        i["id"] = f"eval_task_{counter}"
        new_vals.append(i)
        counter += 1
    for v in new_vals:
        json.dump(v, outer)
        outer.write("\n")