import json
import os
# this script combined the individual category responses into larger groups
rel_size = ["biggest.json", "heaviest.json", "fits.json", "interact.json"]
can_do = ["can_do.json", "can_do_size.json", "can_do_shape.json", "can_do_char.json", "can_do_goal.json"]
is_a_dif = ["difference.json", "diff_criteria.json", "use_as.json","is_a.json", "types_of.json"]
risky = ["injury.json","danger.json", "damage_to_obj.json"]
equip = ["explain_use.json", "equip_used.json", "equip_in_task.json"]
obj_facts = ["obj_loc.json","objs_in_loc.json", "secondary_use.json"]
earthquake = ["earthquake.json"]
instr = ["instruct.json", "followup.json"]

catos = {"rel_size": rel_size, "can_do": can_do, "is_a_dif": is_a_dif, "risky": risky, "equip": equip,\
          "obj_facts": obj_facts, "earthquake": earthquake, "instr": instr}

out_dir = "../gemini_results/"
for key, val in catos.items():
    go_out = []
    for v in val:
        curr_file = json.load(open(os.path.join(out_dir, v)))
        go_out = go_out + curr_file
    out = open(os.path.join(out_dir, f"{key}.json"),"w")
    json.dump(go_out, out)


