# This script is to get the average reasonableness and informativeness scores
# from disagree.agreements import BiDisagreements
import pandas as pd
from metrics import Metrics, Krippendorff

def write_averages(list, type, out):
    # int_a = [int(i) for i in list['a']]
    int_c = [int(i) for i in list['c']]
    int_t = [int(i) for i in list['t']]
    int_all = int_c + int_t
    out.write(f"Average {type} value overall: {sum(int_all)/len(int_all)}\n")
    out.write(f"Average {type} value Taylor: {sum(int_t)/len(int_t)}\n")
    out.write(f"Average {type} value Claire: {sum(int_c)/len(int_c)}\n\n")
    # out.write(f"Average {type} value Austin: {sum(int_a)/len(int_a)}\n\n")

def write_single_avgs(filename, out, name):
    inf = []
    reas = []
    q_inf= []
    with open(filename) as input:
        for line in input:
            vals = line.split(',')[1:]
            if vals[0] == 'instruction' or vals[0] == 'Reasonable':
                continue
            if vals[-1] == '\n':
                vals.pop()
            if len(vals) <= 2:
                vals.append('0')
            ri = 1
            ii = 2
            reas.append(vals[ri])
            inf.append(vals[ii])
            if vals[ri] == '1':
                q_inf.append(vals[ii])
    int_inf = [int(i) for i in inf]
    int_q_inf = [int(i) for i in q_inf]
    int_reas = [int(i) for i in reas]
    outie.write(f"number examples annotated by {name}: {len(int_reas)}")
    out.write(f"Average inform score, all values {name}: {sum(int_inf)/len(int_inf)}\n")
    out.write(f"Average inform score, reasonable values only {name}: {sum(int_q_inf)/len(int_q_inf)}\n")
    out.write(f"Average reason score {name}: {sum(int_reas)/len(int_reas)}\n\n")
            

inf = {'c':[],'t':[]}
inf_if_reas = {'c':[],'t':[]}
reas = {'c':[],'t':[]}

# open("qual_check_all-MS.csv") as a,
with open("qual_check_all-CB.csv") as c, open("qual_check_all-TH.csv") as t:
    for c_line, t_line in zip(c,t):
        # a_vals = a_line.split(',')[1:]
        c_vals = c_line.split(',')[1:]
        t_vals = t_line.split(',')[1:]

        if c_vals[-1] == '\n':
            c_vals.pop()
        # if a_vals[-1] == '\n':
        #     a_vals.pop()
        if t_vals[-1] == '\n':
            t_vals.pop()
    
        if c_vals[0] == "instruction":
            continue
        # if len(a_vals) == 2:
        #     a_vals.append('0')
        if len(c_vals) == 2:
            c_vals.append('0')
        if len(t_vals) == 2:
            t_vals.append('0')

        # reas['a'].append(a_vals[1].strip('\n'))
        # inf['a'].append(a_vals[2].strip('\n'))
        reas['c'].append(c_vals[1].strip('\n'))
        inf['c'].append(c_vals[2].strip('\n'))      
        reas['t'].append(t_vals[1].strip('\n'))
        inf['t'].append(t_vals[2].strip('\n'))

        if int(c_vals[1]) == 1 or int(t_vals[1]) == 1:
            if c_vals[1] == '1' and t_vals[1] == '1':
                inf_if_reas['c'].append(c_vals[2].strip('\n')) 
                inf_if_reas['t'].append(t_vals[2].strip('\n'))
            elif c_vals[1] == '1':
                inf_if_reas['c'].append(c_vals[2].strip('\n')) 
                inf_if_reas['t'].append(None)
            else:
                inf_if_reas['t'].append(t_vals[2].strip('\n'))
                inf_if_reas['c'].append(None) 

inf_pd = pd.DataFrame(inf)
reas_pd = pd.DataFrame(reas)
inf_reas_pd = pd.DataFrame(inf_if_reas)
inf_data = Metrics(inf_pd)
reas_data = Metrics(reas_pd)
inf_reas_data = Metrics(inf_reas_pd)
inf_krip = Krippendorff(inf_pd)
reas_krip = Krippendorff(reas_pd)
# inf_reas_krip = Krippendorff(inf_reas_pd)
with open("human_stats.txt","w") as outie:
    rk = reas_krip.alpha(data_type="nominal")
    outie.write(f"Krippendorf Alpha reason:{rk}\n")
    # outie.write(f"naive kappa reason: {calc_reas.kappa()}")
    # ack = reas_data.cohens_kappa('a','c')
    # atk = reas_data.cohens_kappa('a','t')
    ctk = reas_data.cohens_kappa('t','c')
    # outie.write(f"cohen's kappa agreement a c reason: {ack}\n")
    # outie.write(f"cohen's kappa agreement a t reason: {atk}\n")
    outie.write(f"cohen's kappa agreement c t reason: {ctk}\n")
    outie.write(f"Fleiss Kappa reason: {reas_data.fleiss_kappa()}\n\n")

    ik = inf_krip.alpha(data_type="nominal")
    outie.write(f"Krippendorf Alpha inform:{ik}\n")
    # ack = inf_data.cohens_kappa('a','c')
    # atk = inf_data.cohens_kappa('a','t')
    ctk = inf_data.cohens_kappa('t','c')  
    # outie.write(f"cohen's kappa agreement a c inform: {ack}\n")
    # outie.write(f"cohen's kappa agreement a t inform: {atk}\n")
    outie.write(f"cohen's kappa agreement c t inform: {ctk}\n") 
    outie.write(f"Fleiss Kappa inform: {inf_data.fleiss_kappa()}\n\n") 

    # irk = inf_reas_krip.alpha(data_type="nominal")
    # outie.write(f"Krippendorf Alpha inform if reason:{irk}\n")
    # ack = inf_data.cohens_kappa('a','c')
    # atk = inf_data.cohens_kappa('a','t')
    ctk = inf_reas_data.cohens_kappa('t','c')  
    # outie.write(f"cohen's kappa agreement a c inform: {ack}\n")
    # outie.write(f"cohen's kappa agreement a t inform: {atk}\n")
    outie.write(f"cohen's kappa agreement c t inform: {ctk}\n") 
    outie.write(f"Fleiss Kappa inform: {inf_reas_data.fleiss_kappa()}\n\n") 
    
    outie.write(f"number of questions of all: {len(reas['c'])}")
    write_averages(inf, "inform", outie)
    write_averages(reas, "reason", outie)
    # write_averages(inf_if_reas, "inform if reason", outie)
    write_single_avgs("qual_check_claire.csv", outie, 'Claire')
    write_single_avgs("qual_check_taylor.csv", outie, 'Taylor')
    # write_single_avgs("qual_check_austin.csv", outie, 'Austin')

   



        