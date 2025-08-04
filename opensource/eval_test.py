import os
import subprocess
import argparse
from multiprocessing import Process
from multiprocessing import Pool, TimeoutError
import multiprocessing

def worker_execute(args):
    tm_id = args[0]
    model_id = args[1]
    drl_eval_res_folder = args[2]
    differentiation_str = args[3]
    graph_topology_name = args[4]
    general_dataset_folder = args[5]
    specific_dataset_folder = args[6]
    rate = args[7]
    r1=args[8]
    r2=args[9]

    subprocess.call(["python script_flow.py -t "+str(tm_id)+" -m "+str(model_id)+" -g "+graph_topology_name+" -o "+drl_eval_res_folder+" -d "+differentiation_str+ ' -f ' + general_dataset_folder + ' -f2 '+specific_dataset_folder + ' -r ' + str(rate)+' -r1 '+ str(r1)+ ' -r2 '+str(r2)], shell=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parse file and create plots')

    parser.add_argument('-d', help='logs data file', type=str, required=True, nargs='+')
    parser.add_argument('-f1', help='Dataset name within dataset_sing_top', type=str, required=True, nargs='+')
    parser.add_argument('-f2', help='specific dataset folder name of the topology to evaluate on', type=str, required=True, nargs='+')
    parser.add_argument('-max_edge', help='maximum number of edges the topology can have', type=int, required=True, nargs='+')
    parser.add_argument('-min_edge', help='minimum number of edges the topology can have', type=int, required=True, nargs='+')
    parser.add_argument('-max_nodes', help='minimum number of nodes the topology can have', type=int, required=True, nargs='+')
    parser.add_argument('-min_nodes', help='minimum number of nodes the topology can have', type=int, required=True, nargs='+')
    parser.add_argument('-n', help='number of processes to use for the pool (number of DEFO instances running at the same time)', type=int, required=True, nargs='+')
    parser.add_argument('-r1', help='link ratio', type=str, required=True, nargs='+')
    parser.add_argument('-r2', help='flow ratio', type=str, required=True, nargs='+') 

    args = parser.parse_args()

    aux = args.d[0].split(".")

    aux = aux[1].split("exp")
    
    differentiation_str = str(aux[1].split("Logs")[0])

    general_dataset_folder = "../srh_datasets/"+args.f1[0]+"/"+args.f2[0]+"/"

    drl_eval_res_folder = "../srh_datasets/"+args.f1[0]+"/evalRes_"+args.f2[0]+"/"

    if not os.path.exists("./Images"):
        os.makedirs("./Images")

    if not os.path.exists(drl_eval_res_folder):
        os.makedirs(drl_eval_res_folder)

    if not os.path.exists(drl_eval_res_folder+differentiation_str):
        os.makedirs(drl_eval_res_folder+differentiation_str)

    x = []
    for i in range(10):
        i=(i+1)/10
        x.append(i)

    x=[0.3]

    linkratio=float(args.r1[0])
    flowratio=float(args.r2[0])
    model_id = 0
    with open(args.d[0]) as fp:
        for line in reversed(list(fp)):
            arrayLine = line.split(":")
            if arrayLine[0]=='MAX REWD':
                model_id = int(arrayLine[2].split(",")[0])
                break
    for ind in x:
        print(f'start rate = {ind}')
        for subdir, dirs, files in os.walk(general_dataset_folder):
            for file in files:
                if file.endswith((".graph")):
                    topology_num_nodes = 0
                    with open(general_dataset_folder+file) as fd:

                        while (True):
                            line = fd.readline()
                            if (line == ""):
                                break
                            if (line.startswith("NODES")):
                                topology_num_nodes = int(line.split(' ')[1])

                            # If we are inside the range of number of nodes
                            if topology_num_nodes>=args.min_nodes[0] and topology_num_nodes<=args.max_nodes[0]:
                                if (line.startswith("EDGES")):
                                    topology_num_edges = int(line.split(' ')[1])
                                    # If we are inside the range of number of edges
                                    if topology_num_edges<=args.max_edge[0] and topology_num_edges>=args.min_edge[0]:
                                        topology_Name = file.split('.')[0]
                                        print("*****")
                                        print("***** Evaluating on file: "+file+" with number of edges "+str(topology_num_edges))
                                        print("*****")                                    
                                        argums = [(tm_id, model_id, drl_eval_res_folder, differentiation_str, topology_Name, general_dataset_folder, args.f2[0], ind, linkratio, flowratio) for tm_id in range(50)]

                                        with Pool(processes=args.n[0]) as pool:
                                            pool.map(worker_execute, argums)
                            else:
                                break

    