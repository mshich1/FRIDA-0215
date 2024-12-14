import numpy as np
import matplotlib.pyplot as plt
import re

with open("../llama_results/sem.txt") as l_in, open("../gemini_results/base_sem.txt") as b_in:
    scores_l = [l for l in l_in]
    scores_b = [b for b in b_in]
    scores = scores_l + scores_b
    ems = {"overall":[], "rel_size":[],"can_do_it": [], "is_a_dif":[],"risky":[],\
           "equip":[], "obj_facts":[], "quake":[], "instr": []}
    for p in scores:
        cat_nums = re.match("([a-zA-Z_]+) average sem score: (0\.\d+|1.0)", p)
        if cat_nums is None:
            continue
        cat = cat_nums.group(1)
        num = float(cat_nums.group(2))
        ems[cat].append(num)
    
    x_axis_labels = ["aFRIDA 8B: relative size","aFRIDA 8B: object function","aFRIDA 8B: differences","aFRIDA 8B: objects causing harm",\
                     "aFRIDA 8B: specialized equipment","aFRIDA 8B: non-functional object facts","aFRIDA 8B: earthquake","aFRIDA 8B: instruction understanding",\
                     "FRIDA 8B", "LLaMa 3.1 8B instruct"]
    y_axis_labels = ["relative size templates", "object function templates", "differences templates", "objects causing harm templates",\
                     "specialized equipment templates", "non-functional object facts templates", "earthquake templates", "instruction understanding templates", "all evaluation data"]

    results = np.array([np.array(ems["rel_size"]),
                       np.array(ems["can_do_it"]),
                       np.array(ems["is_a_dif"]),
                       np.array(ems["risky"]),
                       np.array(ems["equip"]),
                       np.array(ems["obj_facts"]),
                       np.array(ems["quake"]),
                       np.array(ems["instr"]),
                       np.array(ems["overall"])])
    
    fig, ax = plt.subplots()
    im = ax.imshow(results)
    plt.rcParams.update({'font.size': 12})
    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(len(x_axis_labels)), labels=x_axis_labels)
    ax.set_yticks(np.arange(len(y_axis_labels)), labels=y_axis_labels)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(y_axis_labels)):
        for j in range(len(x_axis_labels)):
            text = ax.text(j, i, f'{results[i, j]:.2f}',
                        ha="center", va="center", color="w")

    ax.set_title("Embedding Vector Cosine Similarity on data subsets")
    fig.tight_layout()
    plt.show()

