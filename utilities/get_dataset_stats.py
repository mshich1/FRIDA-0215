import json
# import mathplotlib.pyplot as plt
import numpy as np
# This script gets the max rouge score for each instruction and the instruction length
# It then generates averages over template dataset, category dataset, and overall dataset
cat_map = {"rel_size":["biggest", "heaviest", "fits", "interact"],\
            "can_do_it":["can_do", "can_do_size", "can_do_shape", "can_do_char", "can_do_goal"], \
            "is_a_dif": ["difference", "diff_criteria", "use_as","is_a", "types_of"], \
            "risky":["injury","danger", "damage_to_obj"], \
            "equip":["explain_use", "equip_used", "equip_in_task"], \
            "obj_facts":["obj_loc","objs_in_loc", "secondary_use"], \
            "quake":["earthquake"], \
            "instr":["instruct", "followup"]}

cat_eval = {"rel_size":{'instr_len':[],'ans_len':[],'rouge_scores':[]},"can_do_it":{'instr_len':[],'ans_len':[],'rouge_scores':[]}, \
            "is_a_dif": {'instr_len':[],'ans_len':[],'rouge_scores':[]}, "risky":{'instr_len':[],'ans_len':[],'rouge_scores':[]}, \
            "equip":{'instr_len':[],'ans_len':[],'rouge_scores':[]}, "obj_facts":{'instr_len':[],'ans_len':[],'rouge_scores':[]},\
            "quake":{'instr_len':[],'ans_len':[],'rouge_scores':[]}, "instr":{'instr_len':[],'ans_len':[],'rouge_scores':[]}}

files= ["biggest", "heaviest", "fits", "interact",
            "can_do", "can_do_size", "can_do_shape", "can_do_char", "can_do_goal", \
            "difference", "diff_criteria", "use_as","is_a", "types_of", \
            "injury","danger", "damage_to_obj", \
            "explain_use", "equip_used", "equip_in_task", \
            "obj_loc","objs_in_loc", "secondary_use", \
            "earthquake", \
            "instruct", "followup"]

def add_to_dict(n,i,c,r):
    cat_eval[n]['instr_len'].append(i)
    cat_eval[n]['ans_len'].append(c)
    cat_eval[n]['rouge_scores'].append(r)

def put_in_cat(filename, i, c, r):
    if filename in cat_map["rel_size"]:
        add_to_dict("rel_size",i,c,r)
    elif filename in cat_map["can_do_it"]:
        add_to_dict("can_do_it",i,c,r)
    elif filename in cat_map["is_a_dif"]:
        add_to_dict("is_a_dif",i,c,r)
    elif filename in cat_map["risky"]:
        add_to_dict("risky",i,c,r)
    elif filename in cat_map["equip"]:
        add_to_dict("equip",i,c,r)
    elif filename in cat_map["obj_facts"]:
        add_to_dict("obj_facts",i,c,r)
    elif filename in cat_map["quake"]:
        add_to_dict("quake",i,c,r)
    elif filename in cat_map["instr"]:
        add_to_dict("instr",i,c,r)
    else:
        print("Freakout")

with open("dataset_stats.txt","w") as stat_out:
    for f in files:
        overall_len = []
        overall_rouge = []
        with open(f"../gemini_results/{f}.json") as data_in:
            all_data = json.load(data_in)
            instr_len = []
            ans_len = []
            rouge_scores = []
            for q in all_data:
                i_len = len(q["instruction"].split(" "))
                instr_len.append(i_len)
                if q['input'] != None:
                    ch_len = len(q["input"].split(" "))
                else:
                    ch_len = 0
                ans_len.append(ch_len)
                overall_len.append(i_len+ch_len)
                r_score = next(iter(q["most_similar_instructions"].values()))#thank you chatGPT for this line specifically
                rouge_scores.append(r_score)
                overall_rouge.append(r_score)
                put_in_cat(f,i_len, ch_len, r_score) 
            np_inst = np.array(instr_len)
            np_ans = np.array(ans_len)
            np_rouge = np.array(rouge_scores)
            stat_out.write(f"**DATASET IS {f}.json**\n")
            stat_out.write(f'average instruction length: {np_inst.mean()}\n')
            stat_out.write(f'median instruction length: {np.median(np_inst)}\n')
            stat_out.write(f'average answer choice length: {np_ans.mean()}\n')
            stat_out.write(f'median answer choice length: {np.median(np_ans)}\n')
            stat_out.write(f'average ROUGE score: {np_rouge.mean()}\n')
            stat_out.write(f'median ROUGE score: {np.median(np_rouge)}\n\n')

    #stats for the categories
    for n,c in cat_eval.items():
        np_inst = np.array(c['instr_len'])
        np_ans = np.array(c['ans_len'])
        np_rouge = np.array(c['rouge_scores'])
        stat_out.write(f'**CATEGORY IS {n}**\n')
        stat_out.write(f'average instruction length: {np_inst.mean()}\n')
        stat_out.write(f'median instruction length: {np.median(np_inst)}\n')
        stat_out.write(f'average answer choice length: {np_ans.mean()}\n')
        stat_out.write(f'median answer choice length: {np.median(np_ans)}\n')
        stat_out.write(f'average ROUGE score: {np_rouge.mean()}\n')
        stat_out.write(f'median ROUGE score: {np.median(np_rouge)}\n\n')   

    #overall stats
    np_ol = np.array(overall_len)
    np_or = np.array(overall_rouge)
    stat_out.write(f'overall avg instruction length: {np_ol.mean()}\n')
    stat_out.write(f'overall median instruction length: {np.median(np_ol)}\n')
    stat_out.write(f'overall avg ROUGE score: {np_or.mean()}\n')
    stat_out.write(f'overall median ROUGE score: {np.median(np_or)}')





