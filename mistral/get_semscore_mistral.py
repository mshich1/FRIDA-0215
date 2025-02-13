import sys

sys.path.insert(0, "/scratch/mshich/gemini-llama-test/llama/semscore/")

import os
import json
from semscore import EmbeddingModelWrapper
from statistics import mean

eval_qs = [json.loads(l) for l in open("../seed_data/seed_tasks_eval.jsonl")]
eval_ans = [i["instances"][0]["output"] for i in eval_qs]

adapter_result_path = "../mistral_results/"
adapter_results_names = ["rel_size","can_do_it","is_a_dif","risky","equip","obj_facts","quake","instr","all","mistral"]

cat_map = {"rel_size":["biggest", "heaviest", "fits", "interact"],\
            "can_do_it":["can_do", "can_do_size", "can_do_shape", "can_do_char", "can_do_goal"], \
            "is_a_dif": ["difference", "diff_criteria", "use_as","is_a", "types_of"], \
            "risky":["injury","danger", "damage_to_obj"], \
            "equip":["explain_use", "equip_used", "equip_in_task"], \
            "obj_facts":["obj_loc","objs_in_loc", "secondary_use"], \
            "quake":["earthquake"], \
            "instr":["instruct", "followup"]}
cat_eval = {"rel_size":[],"can_do_it":[], "is_a_dif": [], "risky":[], "equip":[], "obj_facts":[], "quake":[], "instr":[]}

def append_to_store(check, i, add_to):
    if check in cat_map["rel_size"]:
        add_to["rel_size"].append(i)
    elif check in cat_map["can_do_it"]:
        add_to["can_do_it"].append(i)
    elif check in cat_map["is_a_dif"]:
        add_to["is_a_dif"].append(i)
    elif check in cat_map["risky"]:
        add_to["risky"].append(i)
    elif check in cat_map["equip"]:
        add_to["equip"].append(i)
    elif check in cat_map["obj_facts"]:
        add_to["obj_facts"].append(i)
    elif check in cat_map["quake"]:
        add_to["quake"].append(i)
    elif check in cat_map["instr"]:
        add_to["instr"].append(i)
    else:
        print(f"failed with following check: {check} and i {i} values")
    return add_to

for l in eval_qs:
    cat_eval = append_to_store(l["cat"], l["instances"][0]["output"], cat_eval)


with open("../mistral_results/sem_mis.txt","w") as outie:
    for a in adapter_results_names:
        cat_results = {"rel_size":[],"can_do_it":[], "is_a_dif": [], "risky":[], "equip":[], "obj_facts":[], "quake":[], "instr":[]}
        path = os.path.join(adapter_result_path, a)
        mod_results = []
        if a != "mistral":
            mod_results = [l.strip() for l in open(f"{path}_mistral.txt")]
        else:
            mod_results = [l.strip() for l in open(f"{path}.txt")]
        for l, m in zip(eval_qs, mod_results):
            cat_results = append_to_store(l["cat"], m, cat_results)
        em = EmbeddingModelWrapper(model_path="mistralai/Ministral-8B-Instruct-2410")
        all_sem = mean(em.get_similarities(em.get_embeddings(mod_results),em.get_embeddings(eval_ans)))
        accs = {}
        for k, v in cat_results.items():
            accs[k] = mean(em.get_similarities(em.get_embeddings(v),em.get_embeddings(cat_eval[k])))
        outie.write(f"***MODEL IS {a}***\n")
        outie.write(f"overall average sem score: {all_sem}\n")
        for k,v in accs.items():
            outie.write(f"{k} average sem score: {v}\n")
        outie.write("\n")
