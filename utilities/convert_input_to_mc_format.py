import json

# This is a script that takes old data from the DMR work and adds in multiple choice formatting 
# ie A) B) C) in front of answer choices. It does not do this for the answer key, I did that by hand like a chump
instruct = [json.loads(l) for l in open("../seed_tasks_earthquake_mc.jsonl","r")]

with open("../seed_tasks_earthquake_mc.jsonl") as input:
    with open("../seed_tasks_earthquake_mc_formatted.jsonl","w") as output:
        count = 0
        for i, start in zip(instruct, input):
            old_input = i["instances"][0]["input"]
            if old_input != "" and count > 2:
                ans_choices = old_input.split(", ")
                new_input = ""
                if len(ans_choices) == 5:
                    new_input = f"A) {ans_choices[0]}\tB) {ans_choices[1]}\tC) {ans_choices[2]}\nD) {ans_choices[3]}\tE) {ans_choices[4]}"
                if len(ans_choices) == 2:
                    new_input = f"A) {ans_choices[0]}\tB) {ans_choices[1]}"
                i["instances"][0]["input"] = new_input
            count += 1
            output.write(f"{i}\n")