#!/usr/bin/python3

import numpy as np
import re
import sys
import networkx as nx

node_to_index_dic = {}
index_to_node_lst = []

def index_to_node(n):
    return(index_to_node_lst[n])

def node_to_index(node):
    return(node_to_index_dic[node])


class Defo_results:
    
    net_size = 0
    MP_matrix = None
    ecmp_routing_matrix = None
    routing_matrix = None
    links_bw = None
    links_weight = None
    Gbase = None
    node_to_index_dic_pvt = None
    index_to_node_lst_pvt = None
    pre_optim_max_load_link = None
    post_optim_max_load_link = None
    
    def __init__(self, graph_file, results_file):
        self.graph_file = graph_file

        self.Gbase = nx.MultiDiGraph()
        self.process_graph_file()

    
    def process_graph_file(self):
        with open(self.graph_file) as fd:
            line = fd.readline()
            camps = line.split(" ")
            self.net_size = int(camps[1])
            # Remove : label x y
            line = fd.readline()
            
            for i in range (self.net_size):
                line = fd.readline()
                node = line[0:line.find(" ")]
                node_to_index_dic[node] = i
                index_to_node_lst.append(node)
                
            self.links_bw = []
            self.links_weight = []
            for i in range(self.net_size):
                self.links_bw.append({})
                self.links_weight.append({})
            for line in fd:
                if (not line.startswith("Link_") and not line.startswith("edge_")):
                    continue
                camps = line.split(" ")
                src = int(camps[1])
                dst = int(camps[2])
                weight = int(camps[3])
                bw = float(camps[4])
                self.Gbase.add_edge(src, dst)
                self.links_bw[src][dst] = bw
                self.links_weight[src][dst] = weight
        self.node_to_index_dic_pvt = node_to_index_dic
        self.index_to_node_lst_pvt = index_to_node_lst

    
    def _get_traffic_matrix (self,traffic_file, tmid):
        tm0bw = np.zeros((self.net_size,self.net_size))
        tm0ps = np.zeros((self.net_size,self.net_size))
        tm0rt = np.zeros((self.net_size,self.net_size))

        tm1bw = np.zeros((self.net_size,self.net_size))
        tm1ps = np.zeros((self.net_size,self.net_size))
        tm1rt = np.zeros((self.net_size,self.net_size))

        tm2ps = np.zeros((self.net_size,self.net_size))
        tm2bw = np.zeros((self.net_size,self.net_size))
        tm2rt = np.zeros((self.net_size,self.net_size))

        ori_tm = np.zeros((self.net_size,self.net_size))

        with open(traffic_file) as fd:
            fd.readline()
            fd.readline()
            for line in fd:
                camps = line.split(" ")
                flow_type = camps[0].split('_')[-1]
                if flow_type == '0':
                    bw = np.floor(float(camps[3]))
                    ps = np.floor(float(camps[4]))
                    rt = bw * 1000 / (ps * 8)
                    tm0bw[int(camps[1]),int(camps[2])] = bw
                    tm0ps[int(camps[1]),int(camps[2])] = ps
                    tm0rt[int(camps[1]),int(camps[2])] = rt

                elif flow_type == '1':
                    bw = np.floor(float(camps[3]))
                    ps = np.floor(float(camps[4]))
                    rt = bw * 1000 / (ps * 8)
                    tm1bw[int(camps[1]),int(camps[2])] = bw
                    tm1ps[int(camps[1]),int(camps[2])] = ps
                    tm1rt[int(camps[1]),int(camps[2])] = rt
                    
                elif flow_type == '2':
                    bw = np.floor(float(camps[3]))
                    ps = np.floor(float(camps[4]))
                    rt = bw * 1000 / (ps * 8)
                    tm2bw[int(camps[1]),int(camps[2])] = bw
                    tm2ps[int(camps[1]),int(camps[2])] = ps
                    tm2rt[int(camps[1]),int(camps[2])] = rt

                    ori_tm[int(camps[1]),int(camps[2])]= tm0bw[int(camps[1]),int(camps[2])]+tm1bw[int(camps[1]),int(camps[2])]+tm2bw[int(camps[1]),int(camps[2])]
        if tmid !=-1:

            return ori_tm, tm0bw, tm0ps, tm0rt, tm1bw, tm1ps, tm1rt, tm2bw, tm2ps, tm2rt
        else:
            return [ori_tm, tm0bw, tm0ps, tm0rt, tm1bw, tm1ps, tm1rt, tm2bw, tm2ps, tm2rt]
                    
    