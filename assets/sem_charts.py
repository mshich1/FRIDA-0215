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

    x_axis_labels = ["size\nmodel","can do\nmodel","dif and\nhypernym model","object risk\nmodel",\
                     "equipment\nmodel","object facts\nmodel","quake\nmodel","instruction id\nmodel",\
                     "all synth\ndata model", "gemini", "llama3.1\ninstruct"]
    x = np.arange(len(x_axis_labels))
    plt.rc('xtick', labelsize = 10)
    # Disaster knowledge - 2nd to last index, value = 2.566666666666667,
    for k,v in ems.items():
        winning_scores = v
         # Figure Size
        fig, ax = plt.subplots(figsize =(16, 9))

        # Horizontal Bar Plot
        ax.barh(x_axis_labels, winning_scores, color = 'coral')

        # Remove axes splines
        for s in ['top', 'bottom', 'left', 'right']:
            ax.spines[s].set_visible(False)

        # Remove x, y Ticks
        ax.xaxis.set_ticks_position('none')
        ax.yaxis.set_ticks_position('none')

        # Add padding between axes and labels
        ax.xaxis.set_tick_params(pad = 5)
        ax.yaxis.set_tick_params(pad = 10)

        # Add x, y gridlines
        ax.grid(visible = True, color ='grey',
                linestyle ='-.', linewidth = 0.5,
                alpha = 0.2)

        # Show top values 
        ax.invert_yaxis()

        # Add annotation to bars
        for i in ax.patches:
            plt.text(i.get_width()+0.05, i.get_y()+0.5, 
                    str(round((i.get_width()), 2)),
                    fontsize = 10, fontweight ='bold',
                    color ='grey')

        # Add Plot Title
        ax.set_title(f'Cosine Similarity of Answer Embeddings for Ablated FRIDA Models',
                    loc ='left', )

        # Show Plot
        if k == 'overall':
            plt.show()

