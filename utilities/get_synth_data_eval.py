import json
import numpy as np
import csv

# This script was to make csv files for human evaluation of the synthetic data
# It randomly samples 10 instructions per template

files= ["biggest", "heaviest", "fits", "interact",
            "can_do", "can_do_size", "can_do_shape", "can_do_char", "can_do_goal", \
            "difference", "diff_criteria", "use_as","is_a", "types_of", \
            "injury","danger", "damage_to_obj", \
            "explain_use", "equip_used", "equip_in_task", \
            "obj_loc","objs_in_loc", "secondary_use", \
            "earthquake", \
            "instruct", "followup"]

with open("qual_check_all.csv","w") as all_check, open("qual_check_taylor.csv","w") as taylor_check, \
    open("qual_check_austin.csv","w") as austin_check, open("qual_check_claire.csv","w") as claire_check:
    counter = 0
    header = 'category,instruction,reasonableness,informativeness\n'
    all_check.write(header)
    taylor_check.write(header)
    austin_check.write(header)
    claire_check.write(header)
    for f in files:
            with open(f"../gemini_results/{f}.json") as data_in:
                all_data = json.load(data_in)
                lines_from_file = np.random.choice(all_data, 10, replace=False)
                lines_from_file = lines_from_file.tolist()
                json_lines = json.loads(json.dumps(lines_from_file))
                lines_from_file = []
                for j in json_lines:
                    j.pop("most_similar_instructions")
                    j.pop("avg_similarity_score")
                    j["instruction"] = j["instruction"].replace(',','')
                    j["output"] = j["output"].replace(',','')
                    if j['input'] != None:
                        j["input"] = j["input"].replace("\t", " ")
                        j["input"] = j["input"].replace("\n", " ")
                        j["input"] = j["input"].replace(",","")
                    lines_from_file.append(f'{f},{j["instruction"]} {j["input"]} Answer: {j["output"]}')
                    #  print(f'{j["instruction"]} {j["input"]} Answer: {j["output"]}')
                lines_for_all = np.random.choice(lines_from_file, 2, replace=False)
                for i in lines_for_all:
                    lines_from_file.remove(i)
                for l in lines_for_all:
                    all_check.write(l)
                    all_check.write(',\n')
                splitup = np.split(lines_from_file, [3, 6])
                if counter%3 == 0:
                    for i in splitup[0]:
                        taylor_check.write(i)
                        taylor_check.write(',\n')
                    for i in splitup[1]:
                        austin_check.write(i)
                        austin_check.write(',\n')
                    for i in splitup[2]:
                         claire_check.write(i)
                         claire_check.write(',\n')
                elif counter%3 == 1: 
                    for i in splitup[0]:
                        claire_check.write(i)
                        claire_check.write(',\n')
                    for i in splitup[1]:
                        austin_check.write(i)
                        austin_check.write(',\n') 
                    for i in splitup[2]:
                        taylor_check.write(i)
                        taylor_check.write(',\n')
                else:
                    for i in splitup[0]:
                        claire_check.write(i)
                        claire_check.write(',\n')
                    for i in splitup[1]:
                        taylor_check.write(i)
                        taylor_check.write(',\n')
                    for i in splitup[2]:
                        austin_check.write(i)
                        austin_check.write(',\n')  
                counter += 1