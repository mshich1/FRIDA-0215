# This script is to get the average reasonableness and informativeness scores
# from disagree.agreements import BiDisagreements
import pandas as pd
from disagree import Metrics, Krippendorff

inf = {'a':[],'c':[],'t':[]}
reas = {'a':[],'c':[],'t':[]}

with open("qual_check_all-AB.csv") as a, open("qual_check_all-CB.csv") as c, open("qual_check_all-TH.csv") as t:
    for a_line, c_line, t_line in zip(a,c,t):
        a_vals = a_line.split(',')[1:]
        c_vals = c_line.split(',')[1:]
        t_vals = t_line.split(',')[1:]

        if c_vals[-1] == '\n':
            c_vals.pop()
        if a_vals[-1] == '\n':
            a_vals.pop()
        if t_vals[-1] == '\n':
            t_vals.pop()
    
        if a_vals[0] == "instruction":
            continue
        if len(a_vals) == 2:
            a_vals.append('0')
        if len(c_vals) == 2:
            c_vals.append('0')
        if len(t_vals) == 2:
            t_vals.append('0')

        reas['a'].append(a_vals[1].strip('\n'))
        inf['a'].append(a_vals[2].strip('\n'))
        reas['c'].append(c_vals[1].strip('\n'))
        inf['c'].append(c_vals[2].strip('\n'))      
        reas['t'].append(t_vals[1].strip('\n'))
        inf['t'].append(t_vals[2].strip('\n'))
inf_pd = pd.DataFrame(inf)
reas_pd = pd.DataFrame(reas)
inf_data = Metrics(inf_pd)
reas_data = Metrics(reas_pd)
inf_krip = Krippendorff(inf_pd)
reas_krip = Krippendorff(reas_pd)
with open("human_stats.txt","w") as outie:
    rk = reas_krip.alpha(data_type="nominal")
    outie.write(f"Krippendorf Alpha reason:{rk}\n")
    # outie.write(f"naive kappa reason: {calc_reas.kappa()}")
    temp = reas_data.kappa('a','c')
    outie.write(f"Average Agreement a c reason: {temp}")
    ik = inf_krip.alpha(data_type="nominal")
    outie.write(f"Krippendorf Alpha inform:{ik}\n")
    # outie.write(f"naive kappa inform: {calc_inf.kappa()}")
    # outie.write(f"Average Observed Agreement inform: {calc_inf.avg_Ao()}")
    def write_averages(list, type):
        # c_list = [i for i in list if i[0]=='c']
        # for i in range(len(c_list)):
        #     print(f"t[2] at index {i}: {c_list[i][2]}")
        t_vals = [int(i[1]) for i in list if i[0] == "t"]
        c_vals = [int(i[1]) for i in list if i[0] == "c"]
        a_vals = [int(i[1]) for i in list if i[0] == "a"]
        all_vals = [int(i[1]) for i in list]
        outie.write(f"Average {type} value overall: {sum(all_vals)/len(all_vals)}\n")
        outie.write(f"Average {type} value Taylor: {sum(t_vals)/len(t_vals)}\n")
        outie.write(f"Average {type} value Claire: {sum(c_vals)/len(c_vals)}\n")
        outie.write(f"Average {type} value Austin: {sum(a_vals)/len(a_vals)}\n\n")
    write_averages(inf, "inform")
    write_averages(reas, "reason")



        