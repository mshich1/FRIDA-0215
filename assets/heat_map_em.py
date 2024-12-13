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
    
    x_axis_labels = ["size model","can do model","dif and hypernym model","object risk model",\
                     "equipment model","object facts model","quake model","instruction id model",\
                     "all synth data model", "gemini", "llama3.1 8B instruct"]
    y_axis_labels = ["size templates", "can do templates", "dif and hypernym templates", "object risk templates",\
                     "equipment templates", "object facts templates", "quake templates", "instruction id templates", "all templates"]
    x = np.arange(len(x_axis_labels))
    y = np.arange(len(y_axis_labels))

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

    ax.set_title("Exact match accuracy on data subsets")
    fig.tight_layout()
    plt.show()
