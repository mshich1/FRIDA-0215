import json

with open("../seed_data/seed_tasks_eval2.jsonl","w") as sorty:
     lines = [json.loads(l) for l in open("../seed_data/seed_tasks_eval.jsonl","r")]
     sort_em = sorted(lines, key = lambda d: d["name"])
     for l in sort_em: 
         json.dump(l, sorty)
         sorty.write("\n")