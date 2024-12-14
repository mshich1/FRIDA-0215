import json
import os
from semscore.semscore import EmbeddingModelWrapper
from statistics import mean

eval_qs = [json.loads(l) for l in open("../seed_data/seed_tasks_eval.jsonl")]
eval_ans = [i["instances"][0]["output"] for i in eval_qs]

adapter_result_path = "../llama_results/"
adapter_results_names = ["rel_size","can_do_it","is_a_dif","risky","equip","obj_facts","quake","instr","all"]
cat_map = {"rel_size":["biggest", "heaviest", "fits", "interact"],\
            "can_do_it":["can_do", "can_do_size", "can_do_shape", "can_do_char", "can_do_goal"], \
            "is_a_dif": ["difference", "diff_criteria", "use_as","is_a", "types_of"], \
            "risky":["injury","danger", "damage_to_obj"], \
            "equip":["explain_use", "equip_used", "equip_in_task"], \
            "obj_facts":["obj_loc","objs_in_loc", "secondary_use"], \
            "quake":["earthquake"], \
            "instr":["instruct", "followup"]}
cat_eval = {"rel_size":[],"can_do_it":[], "is_a_dif": [], "risky":[], "equip":[], "obj_facts":[], "quake":[], "instr":[]}

for l in eval_qs:
    if l["cat"] in cat_map["rel_size"]:
        cat_eval["rel_size"].append(l["instances"][0]["output"])
    elif l["cat"] in cat_map["can_do_it"]:
        cat_eval["can_do_it"].append(l["instances"][0]["output"])
    elif l["cat"] in cat_map["is_a_dif"]:
        cat_eval["is_a_dif"].append(l["instances"][0]["output"])
    elif l["cat"] in cat_map["risky"]:
        cat_eval["risky"].append(l["instances"][0]["output"])
    elif l["cat"] in cat_map["equip"]:
        cat_eval["equip"].append(l["instances"][0]["output"])
    elif l["cat"] in cat_map["obj_facts"]:
        cat_eval["obj_facts"].append(l["instances"][0]["output"])
    elif l["cat"] in cat_map["quake"]:
        cat_eval["quake"].append(l["instances"][0]["output"])
    elif l["cat"] in cat_map["instr"]:
        cat_eval["instr"].append(l["instances"][0]["output"])
    else:
        continue

with open("../llama_results/sem_md.txt","w") as outie:
    for a in adapter_results_names:
        cat_results = {"rel_size":[],"can_do_it":[], "is_a_dif": [], "risky":[], "equip":[], "obj_facts":[], "quake":[], "instr":[]}
        path = os.path.join(adapter_result_path, a)
        mod_results = [l.strip() for l in open(f"{path}.txt")]
        # print(f"mod results: {mod_results}")
        for l, m in zip(eval_qs, mod_results):
            if l["cat"] in cat_map["rel_size"]:
                cat_results["rel_size"].append(m)
            elif l["cat"] in cat_map["can_do_it"]:
                cat_results["can_do_it"].append(m)
            elif l["cat"] in cat_map["is_a_dif"]:
                cat_results["is_a_dif"].append(m)
            elif l["cat"] in cat_map["risky"]:
                cat_results["risky"].append(m)
            elif l["cat"] in cat_map["equip"]:
                cat_results["equip"].append(m)
            elif l["cat"] in cat_map["obj_facts"]:
                cat_results["obj_facts"].append(m)
            elif l["cat"] in cat_map["quake"]:
                cat_results["quake"].append(m)
            elif l["cat"] in cat_map["instr"]:
                cat_results["instr"].append(m)
            else:
                continue
        em = EmbeddingModelWrapper(model_path="meta-llama/Llama-3.2-3B-Instruct")
        all_sem = mean(em.get_similarities(em.get_embeddings(mod_results),em.get_embeddings(eval_ans)))
        accs = {}
        for k, v in cat_results.items():
            accs[k] = mean(em.get_similarities(em.get_embeddings(v),em.get_embeddings(cat_eval[k])))
        outie.write(f"***MODEL IS {a}***\n")
        outie.write(f"overall average sem score: {all_sem}\n")
        for k,v in accs.items():
            outie.write(f"{k} average sem score: {v}\n")
        outie.write("\n")

# with open("../gemini_results/base_sem.txt","w") as outie:
#     for a in ["../gemini_results/gemini.txt","../llama_results/llama.txt"]:
#         cat_results = {"rel_size":[],"can_do_it":[], "is_a_dif": [], "risky":[], "equip":[], "obj_facts":[], "quake":[], "instr":[]}
#         mod_results = [l.strip() for l in open(a)]
#         # print(f"mod results: {mod_results}")
#         for l, m in zip(eval_qs, mod_results):
#             if l["cat"] in cat_map["rel_size"]:
#                 cat_results["rel_size"].append(m)
#             elif l["cat"] in cat_map["can_do_it"]:
#                 cat_results["can_do_it"].append(m)
#             elif l["cat"] in cat_map["is_a_dif"]:
#                 cat_results["is_a_dif"].append(m)
#             elif l["cat"] in cat_map["risky"]:
#                 cat_results["risky"].append(m)
#             elif l["cat"] in cat_map["equip"]:
#                 cat_results["equip"].append(m)
#             elif l["cat"] in cat_map["obj_facts"]:
#                 cat_results["obj_facts"].append(m)
#             elif l["cat"] in cat_map["quake"]:
#                 cat_results["quake"].append(m)
#             elif l["cat"] in cat_map["instr"]:
#                 cat_results["instr"].append(m)
#             else:
#                 continue
#         em = EmbeddingModelWrapper(model_path="meta-llama/Meta-Llama-3.1-8B-Instruct")
#         all_sem = mean(em.get_similarities(em.get_embeddings(mod_results),em.get_embeddings(eval_ans)))
#         accs = {}
#         for k, v in cat_results.items():
#             accs[k] = mean(em.get_similarities(em.get_embeddings(v),em.get_embeddings(cat_eval[k])))
#         outie.write(f"***MODEL IS {a}***\n")
#         outie.write(f"overall average sem score: {all_sem}\n")
#         for k,v in accs.items():
#             outie.write(f"{k} average sem score: {v}\n")
#         outie.write("\n")
