import json
import numpy as np

eval_qs = [json.loads(l) for l in open("../seed_data/seed_tasks_eval.jsonl")]

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


for a in adapter_results_names:
    cat_results = {"rel_size":[],"can_do_it":[], "is_a_dif": [], "risky":[], "equip":[], "obj_facts":[], "quake":[], "instr":[]}
