# This script is to get the average reasonableness and informativeness scores

from nltk.metrics.agreement import AnnotationTask
inf = []
reas = []

with open("qual_check_all-AB.csv") as a, open("qual_check_all-CB.csv") as c, open("qual_check_all-TH.csv") as t:
    counter = 0
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

        reas.append(("a",a_vals[1].strip('\n'),a_vals[0]))
        inf.append(("a",a_vals[2].strip('\n'),a_vals[0]))
        reas.append(("c",c_vals[1].strip('\n'),c_vals[0]))
        inf.append(("c",c_vals[2].strip('\n'),c_vals[0]))      
        reas.append(("t",t_vals[1].strip('\n'),t_vals[0]))
        inf.append(("t",t_vals[2].strip('\n'),t_vals[0]))
        counter += 1
calc_reas = AnnotationTask(data=reas)
calc_inf = AnnotationTask(data=inf)
with open("human_stats.txt","w") as outie:
    outie.write(f"Krippendorf Alpha reason:{calc_reas.alpha()}\n")
    # outie.write(f"naive kappa reason: {calc_reas.kappa()}")
    temp = calc_reas.Ao("a","c")
    outie.write(f"Average Agreement a c reason: {temp}")
    outie.write(f"Krippendorf Alpha inform:{calc_inf.alpha()}\n")
    # outie.write(f"naive kappa inform: {calc_inf.kappa()}")
    outie.write(f"Average Observed Agreement inform: {calc_inf.avg_Ao()}")
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



        