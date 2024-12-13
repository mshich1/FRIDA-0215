import numpy as np
import matplotlib.pyplot as plt
import re

with open("../llama_results/em.txt") as l_in, open("../gemini_results/base_em.txt") as b_in:
    scores_l = [l for l in l_in]
    scores_b = [b for b in b_in]
    scores = scores_l + scores_b
    ems = {"overall":[], "rel_size":[],"can_do_it": [], "is_a_dif":[],"risky":[],\
           "equip":[], "obj_facts":[], "quake":[], "instr": []}
    for p in scores:
        cat_nums = re.match("([a-zA-Z_]+) accuracy: {\'exact_match\': (0\.\d+|1.0)}", p)
        if cat_nums is None:
            continue
        cat = cat_nums.group(1)
        num = float(cat_nums.group(2))
        ems[cat].append(num)
    
    x_axis_labels = ["size\nmodel","can do\nmodel","dif and\nhypernym model","object risk\nmodel",\
                     "equipment\nmodel","object facts\nmodel","quake\nmodel","instruction id\nmodel",\
                     "all synth\ndata model", "gemini", "llama3.1\ninstruct"]
    x = np.arange(len(x_axis_labels))
    plt.rc('xtick', labelsize = 10)
    # Disaster knowledge - 2nd to last index, value = 2.566666666666667,
    for k,v in ems.items():
        winning_scores = v
        print(v)
        print(len(v))
        width = 0.25
        plt.bar(x, winning_scores, width, color='cornflowerblue')
        plt.xticks(x, x_axis_labels)
        plt.xlabel("Model")
        plt.ylabel("Exact Match")
        plt.ylim(0,1)
        plt.title(f"LM's Exact Match score for {k} data")
        plt.show()
