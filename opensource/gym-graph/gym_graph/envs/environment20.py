import gym
import numpy as np
import networkx as nx
import random
from gym import error, spaces, utils
from random import choice
import pandas as pd
import pickle
import json 
import os.path
import gc
import opensource.defo_process_results as defoResults
import matplotlib.pyplot as plt
import tensorflow as tf
import time
from itertools import product

class Env20(gym.Env):

    def __init__(self):
        self.flowAmountDict = {}
        self.srnode = None
        self.graph = None 
        self.source = None
        self.destination = None
        self.demand = None
        self.afterAllocateBandDict = {}
        self.edge_state = None
        self.graph_topology_name = None
        self.dataset_folder_name = None 
        self.ori_tm = None
        self.tm0bw=None 
        self.tm0ps=None
        self.tm0rt=None
        self.tm1bw=None
        self.tm1ps=None
        self.tm1rt=None
        self.tm2bw=None 
        self.tm2ps=None
        self.tm2rt = None
        self.tmlist = [self.ori_tm, self.tm0bw, self.tm0ps, self.tm0rt, self.tm1bw, self.tm1ps, self.tm1rt, self.tm2bw, self.tm2ps, self.tm2rt]
        self.diameter = None
        sr_rate = None
        self.packet_len = 1
        self.arrval_dict = {}

        self.first = None
        self.firstTrueSize = None
        self.second = None
        self.between_feature = None

        self.percentage_demands = None
        self.shufle_demands = False 
        self.top_K_critical_demands = False 
        self.num_critical_links = 5

        self.sp_middlepoints = None 
        self.shortest_paths = None 
        self.sp_middlepoints_step = dict() 

        self.srbandConsume = dict()

        self.srpathlength = dict()

        self.mu_bet = None
        self.std_bet = None


        self.episode_length = None
        self.currentVal = None 
        self.initial_maxLinkUti = None
        self.iter_list_elig_demn = None
        self.spath = None


        self.error_evaluation = None

        self.target_link_capacity = None

        self.TM = None 
        self.sumTM = None
        self.routing = None
        self.paths_Matrix_from_routing = None 

        self.K = None
        self.nodes = None 
        self.ordered_edges = None
        self.edgesDict = dict() 
        self.previous_path = None

        self.src_dst_k_middlepoints = None 
        self.list_eligible_demands = None 
        self.link_capacity_feature = None

        self.numNodes = None
        self.numEdges = None
        self.next_state = None


        self.edgeMaxUti = None 

        self.edgeMinUti = None 

        self.patMaxBandwth = None 
        self.maxBandwidth = None

        self.episode_over = True
        self.reward = 0
        self.allPaths = dict() 
        self.sr_rate=None

    def seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
    
    def add_features_to_edges(self):
        incId = 1
        for node in self.graph:
            for adj in self.graph[node]:
                if not 'betweenness' in self.graph[node][adj][0]:
                    self.graph[node][adj][0]['betweenness'] = 0
                if not 'edgeId' in self.graph[node][adj][0]:
                    self.graph[node][adj][0]['edgeId'] = incId
                if not 'numsp' in self.graph[node][adj][0]:
                    self.graph[node][adj][0]['numsp'] = 0
                if not 'utilization' in self.graph[node][adj][0]:
                    self.graph[node][adj][0]['utilization'] = 0
                if not 'capacity' in self.graph[node][adj][0]:
                    self.graph[node][adj][0]['capacity'] = 0
                if not 'weight' in self.graph[node][adj][0]:
                    self.graph[node][adj][0]['weight'] = 0
                if not 'kshortp' in self.graph[node][adj][0]:
                    self.graph[node][adj][0]['kshortp'] = 0
                if not 'crossing_paths' in self.graph[node][adj][0]: 
                    self.graph[node][adj][0]['crossing_paths'] = dict()
                incId = incId + 1

    def num_shortest_path(self, topology):
        self.diameter = nx.diameter(self.graph)

        for n1 in range (0,self.numNodes):
            for n2 in range (0,self.numNodes):
                if (n1 != n2):

                    if str(n1)+':'+str(n2) not in self.allPaths:
                        self.allPaths[str(n1)+':'+str(n2)] = []

                    [self.allPaths[str(n1)+':'+str(n2)].append(p) for p in nx.all_simple_paths(self.graph, source=n1, target=n2, cutoff=self.diameter*2)]


                    self.allPaths[str(n1)+':'+str(n2)] = sorted(self.allPaths[str(n1)+':'+str(n2)], key=lambda item: (len(item), item))
                    path = 0
                    while path < self.K and path < len(self.allPaths[str(n1)+':'+str(n2)]):
                        currentPath = self.allPaths[str(n1)+':'+str(n2)][path]
                        i = 0
                        j = 1


                        while (j < len(currentPath)):
                            self.graph.get_edge_data(currentPath[i], currentPath[j])[0]['numsp'] = \
                                self.graph.get_edge_data(currentPath[i], currentPath[j])[0]['numsp'] + 1
                            i = i + 1
                            j = j + 1

                        path = path + 1


                    del self.allPaths[str(n1)+':'+str(n2)][path:len(self.allPaths[str(n1)+':'+str(n2)])]
                    gc.collect()
    
    def decrease_links_utilization_sp(self, src, dst, init_source, final_destination):

        bw_allocated = self.TM[init_source][final_destination]
        currentPath = self.shortest_paths[src,dst]

        i = 0
        j = 1
        while (j < len(currentPath)):
            firstNode = currentPath[i]
            secondNode = currentPath[j]

            self.graph[firstNode][secondNode][0]['utilization'] -= bw_allocated 
            if str(init_source)+':'+str(final_destination) in self.graph[firstNode][secondNode][0]['crossing_paths']:
                del self.graph[firstNode][secondNode][0]['crossing_paths'][str(init_source)+':'+str(final_destination)]
            self.edge_state[self.edgesDict[str(firstNode)+':'+str(secondNode)]][0] = self.graph[firstNode][secondNode][0]['utilization']
            i = i + 1
            j = j + 1


    def decrease_links_utilization_sp_init(self, src, pdict, init_source, final_destination):

        path = pdict['path']

        srnode = pdict['active_sr']

        if len(srnode) == 0:
            bw_allocate = self.TM[init_source][final_destination]
            while (j < len(path)):
                firstNode = path[i]
                secondNode = path[j]
                

                self.graph[firstNode][secondNode][0]['utilization'] -= bw_allocate  
                if str(init_source)+':'+str(final_destination) in self.graph[firstNode][secondNode][0]['crossing_paths']:
                    del self.graph[firstNode][secondNode][0]['crossing_paths'][str(init_source)+':'+str(final_destination)]
                self.edge_state[self.edgesDict[str(firstNode)+':'+str(secondNode)]][0] = self.graph[firstNode][secondNode][0]['utilization']
                i = i + 1
                j = j + 1
        
        if len(srnode)==1:
            oribw = self.ori_tm[init_source][final_destination]
            srnode0 = srnode[0]
            srnode1 = srnode[1]
            srhlen = (8+16+40)*8/1000
            add_bw = srhlen * (self.tm0rt + self.tm1rt + self.tm2rt)
            new_bw = add_bw+oribw
            flag = 0
            while (j < len(path)):
                firstNode = path[i]
                secondNode = path[j]
                if firstNode != srnode0 and flag != 1:
                    
                    self.graph[firstNode][secondNode][0]['utilization'] -= oribw  
                    if str(init_source)+':'+str(final_destination) in self.graph[firstNode][secondNode][0]['crossing_paths']:
                        del self.graph[firstNode][secondNode][0]['crossing_paths'][str(init_source)+':'+str(final_destination)]
                    self.edge_state[self.edgesDict[str(firstNode)+':'+str(secondNode)]][0] = self.graph[firstNode][secondNode][0]['utilization']
                    i = i + 1
                    j = j + 1
                if secondNode == srnode1:
                    flag = 0
                    self.graph[firstNode][secondNode][0]['utilization'] -= new_bw  
                    if str(init_source)+':'+str(final_destination) in self.graph[firstNode][secondNode][0]['crossing_paths']:
                        del self.graph[firstNode][secondNode][0]['crossing_paths'][str(init_source)+':'+str(final_destination)]
                    self.edge_state[self.edgesDict[str(firstNode)+':'+str(secondNode)]][0] = self.graph[firstNode][secondNode][0]['utilization']
                    i = i + 1
                    j = j + 1
                else:
                    flag = 1
                    self.graph[firstNode][secondNode][0]['utilization'] -= new_bw  
                    if str(init_source)+':'+str(final_destination) in self.graph[firstNode][secondNode][0]['crossing_paths']:
                        del self.graph[firstNode][secondNode][0]['crossing_paths'][str(init_source)+':'+str(final_destination)]
                    self.edge_state[self.edgesDict[str(firstNode)+':'+str(secondNode)]][0] = self.graph[firstNode][secondNode][0]['utilization']
                    i = i + 1
                    j = j + 1

        if len(srnode)==2:
            oribw = self.ori_tm[init_source][final_destination]
            srnode0 = srnode[0]
            srnode1 = srnode[1]
            srnode2 = srnode[2]
            srhlen = (8+32+40)*8/1000
            add_bw = srhlen * (self.tm0rt + self.tm1rt + self.tm2rt)
            new_bw = add_bw+oribw
            flag = 0
            while (j < len(path)):

                firstNode = path[i]
                secondNode = path[j]
                if firstNode != srnode0 and flag != 1:
                    
                    self.graph[firstNode][secondNode][0]['utilization'] -= oribw  
                    if str(init_source)+':'+str(final_destination) in self.graph[firstNode][secondNode][0]['crossing_paths']:
                        del self.graph[firstNode][secondNode][0]['crossing_paths'][str(init_source)+':'+str(final_destination)]
                    self.edge_state[self.edgesDict[str(firstNode)+':'+str(secondNode)]][0] = self.graph[firstNode][secondNode][0]['utilization']
                    i = i + 1
                    j = j + 1
                if secondNode == srnode2:
                    flag = 0
                    self.graph[firstNode][secondNode][0]['utilization'] -= new_bw  
                    if str(init_source)+':'+str(final_destination) in self.graph[firstNode][secondNode][0]['crossing_paths']:
                        del self.graph[firstNode][secondNode][0]['crossing_paths'][str(init_source)+':'+str(final_destination)]
                    self.edge_state[self.edgesDict[str(firstNode)+':'+str(secondNode)]][0] = self.graph[firstNode][secondNode][0]['utilization']
                    i = i + 1
                    j = j + 1
                else:
                    flag = 1
                    self.graph[firstNode][secondNode][0]['utilization'] -= new_bw  
                    if str(init_source)+':'+str(final_destination) in self.graph[firstNode][secondNode][0]['crossing_paths']:
                        del self.graph[firstNode][secondNode][0]['crossing_paths'][str(init_source)+':'+str(final_destination)]
                    self.edge_state[self.edgesDict[str(firstNode)+':'+str(secondNode)]][0] = self.graph[firstNode][secondNode][0]['utilization']
                    i = i + 1
                    j = j + 1

    def _get_top_k_critical_flows(self, list_ids):
        self.list_eligible_demands.clear()
        for linkId in list_ids:
            i = linkId[1]
            j = linkId[2]
            for demand, value in self.graph[i][j][0]['crossing_paths'].items():
                src, dst = int(demand.split(':')[0]), int(demand.split(':')[1])
                if (src, dst, self.TM[src,dst]) not in self.list_eligible_demands:  
                    self.list_eligible_demands.append((src, dst, self.TM[src,dst]))

        self.list_eligible_demands = sorted(self.list_eligible_demands, key=lambda tup: tup[2], reverse=True)
        if len(self.list_eligible_demands)>int(np.ceil(self.numNodes*(self.numNodes-1)*self.percentage_demands)):
            self.list_eligible_demands = self.list_eligible_demands[:int(np.ceil(self.numNodes*(self.numNodes-1)*self.percentage_demands))]

    def compute_centroid(matrices):

        stack = np.stack(matrices, axis=0) 
        
        centroid = np.mean(stack, axis=0)
        return centroid


    def _generate_tm(self, tm_id):
        sumtm=[]
        if tm_id !=-1:

            graph_file = self.dataset_folder_name+"/"+self.graph_topology_name+".graph"

            results_file = self.dataset_folder_name+"/res_"+self.graph_topology_name+"_"+str(tm_id)
            tm_file = self.dataset_folder_name+"/TM/"+self.graph_topology_name+'.'+str(tm_id)+".demands"
            
            self.defoDatasetAPI = defoResults.Defo_results(graph_file,results_file)
            self.links_bw = self.defoDatasetAPI.links_bw
            self.MP_matrix = self.defoDatasetAPI.MP_matrix
            self.ori_tm, self.tm0bw, self.tm0ps, self.tm0rt, self.tm1bw, self.tm1ps, self.tm1rt, self.tm2bw, self.tm2ps, self.tm2rt = self.defoDatasetAPI._get_traffic_matrix(tm_file)
            self.TM=self.ori_tm
        else:
            for i in range(100):

                graph_file = self.dataset_folder_name+"/"+self.graph_topology_name+".graph"

                results_file = self.dataset_folder_name+"/res_"+self.graph_topology_name+"_"+str(i)
                tm_file = self.dataset_folder_name+"/TM/"+self.graph_topology_name+'.'+str(i)+".demands"
                
                self.defoDatasetAPI = defoResults.Defo_results(graph_file,results_file)
                self.links_bw = self.defoDatasetAPI.links_bw

                sumtm = self.defoDatasetAPI._get_traffic_matrix(tm_file,tm_id)
                
            for j in range(len(sumtm)):
                self.tmlist[j]=self.compute_centroid(sumtm[j])
                
        self.iter_list_elig_demn = 0
        self.list_eligible_demands.clear()
        min_links_bw = 1000000.0
        for src in range (0,self.numNodes):
            for dst in range (0,self.numNodes):
                if src!=dst:
                    self.list_eligible_demands.append((src, dst, self.ori_tm[src,dst]))

                    if src in self.graph and dst in self.graph[src]:

                        if self.links_bw[src][dst]<min_links_bw:
                            min_links_bw = self.links_bw[src][dst]
                        

                        self.graph[src][dst][0]['utilization'] = 0.0
                        self.graph[src][dst][0]['crossing_paths'].clear()
        

        if self.shufle_demands:
            random.shuffle(self.list_eligible_demands)
            self.list_eligible_demands = self.list_eligible_demands[:int(np.ceil(len(self.list_eligible_demands)*self.percentage_demands))]
        elif self.top_K_critical_demands:


            self.list_eligible_demands = sorted(self.list_eligible_demands, key=lambda tup: tup[2], reverse=True)
            self.list_eligible_demands = self.list_eligible_demands[:int(np.ceil(len(self.list_eligible_demands)*self.percentage_demands))]

    def compute_link_utilization_reset(self):

        for src in range (0,self.numNodes):
            for dst in range (0,self.numNodes):
                if src!=dst:
                    self.allocate_to_destination_sp(src, dst, src, dst)
    
    def _obtain_path_more_bandwidth_rand_link(self):

        sorted_dict = list((k, v) for k, v in sorted(self.graph[self.edgeMaxUti[0]][self.edgeMaxUti[1]][0]['crossing_paths'].items(), key=lambda item: item[1], reverse=True))
        path = random.randint(0, 1)

        if path>=len(sorted_dict):
            path = 0
        srcPath = int(sorted_dict[path][0].split(':')[0])
        dstPath = int(sorted_dict[path][0].split(':')[1])
        self.patMaxBandwth = (srcPath, dstPath, self.TM[srcPath][dstPath])
    
    def _obtain_path_from_set_rand(self):
        len_demans = len(self.list_eligible_demands)-1
        path = random.randint(0, len_demans)
        srcPath = int(self.list_eligible_demands[path][0])
        dstPath = int(self.list_eligible_demands[path][1])
        self.patMaxBandwth = (srcPath, dstPath, int(self.list_eligible_demands[path][2]))
    
    def _obtain_demand(self):
        src = self.list_eligible_demands[self.iter_list_elig_demn][0]
        dst = self.list_eligible_demands[self.iter_list_elig_demn][1]
        bw = self.list_eligible_demands[self.iter_list_elig_demn][2]
        self.patMaxBandwth = (src, dst, int(bw))
        self.iter_list_elig_demn += 1
    
    def get_value(self, source, destination, action):

        middlePointList = self.src_dst_k_middlepoints[str(source) +':'+ str(destination)]
        middlePoint = middlePointList[action]


        self.allocate_to_destination_sp(source, middlePoint, source, destination)

        if middlePoint!=destination:

            self.allocate_to_destination_sp(middlePoint, destination, source, destination)

            self.sp_middlepoints[str(source)+':'+str(destination)] = middlePoint
        
        currentValue = -1000000

        for i in self.graph:
            for j in self.graph[i]:
                position = self.edgesDict[str(i)+':'+str(j)]
                link_capacity = self.links_bw[i][j]
                if self.edge_state[position][0]/link_capacity>currentValue:
                    currentValue = self.edge_state[position][0]/link_capacity

        if str(source)+':'+str(destination) in self.sp_middlepoints:
            middlepoint = self.sp_middlepoints[str(source)+':'+str(destination)]
            self.decrease_links_utilization_sp(source, middlepoint, source, destination)
            self.decrease_links_utilization_sp(middlepoint, destination, source, destination)
            del self.sp_middlepoints[str(source)+':'+str(destination)] 
        else: 
            self.decrease_links_utilization_sp(source, destination, source, destination)
        
        return -currentValue  

    def _obtain_demand_hill_climbing(self):
        dem_iter = 0
        nextVal = -1000000
        self.next_state = None
   
        for source in range(self.numNodes):
            for dest in range(self.numNodes):
                if source!=dest:
                    for action in range(len(self.src_dst_k_middlepoints[str(source)+':'+str(dest)])):
                        middlepoint = -1
   
   
                        if str(source)+':'+str(dest) in self.sp_middlepoints:
                            middlepoint = self.sp_middlepoints[str(source)+':'+str(dest)]
                            self.decrease_links_utilization_sp(source, middlepoint, source, dest)
                            self.decrease_links_utilization_sp(middlepoint, dest, source, dest)
                            del self.sp_middlepoints[str(source)+':'+str(dest)] 
                        else: 
                            self.decrease_links_utilization_sp(source, dest, source, dest)

                        evalState = self.get_value(source, dest, action)
                        if evalState > nextVal:
                            nextVal = evalState
                            self.next_state = (action, source, dest)

                        if middlepoint>=0:
                           
                            self.allocate_to_destination_sp(source, middlepoint, source, dest)

                            self.allocate_to_destination_sp(middlepoint, dest, source, dest)

                            self.sp_middlepoints[str(source)+':'+str(dest)] = middlepoint
                        else:

                            self.allocate_to_destination_sp(source, dest, source, dest)
        self.patMaxBandwth = (self.next_state[1], self.next_state[2], self.TM[self.next_state[1]][self.next_state[2]])

    def compute_middlepoint_set_random(self):

        self.src_dst_k_middlepoints = dict()

        for n1 in range (0,self.numNodes):
            for n2 in range (0,self.numNodes):
                if (n1 != n2):
                    num_middlepoints = 0
                    self.src_dst_k_middlepoints[str(n1)+':'+str(n2)] = list()

                    self.src_dst_k_middlepoints[str(n1)+':'+str(n2)].append(n2)
                    num_middlepoints += 1
                    while num_middlepoints<self.K:
                        middlpt = np.random.randint(0, self.numNodes)
                        while middlpt==n1 or middlpt==n2 or middlpt in self.src_dst_k_middlepoints[str(n1)+':'+str(n2)]:
                            middlpt = np.random.randint(0, self.numNodes)
                        self.src_dst_k_middlepoints[str(n1)+':'+str(n2)].append(middlpt)
                        num_middlepoints += 1         

        
    def mark_edges(self, action_flags, src, dst, init_source, final_destination):
        currentPath = self.shortest_paths[src,dst]
        
        i = 0
        j = 1

        while (j < len(currentPath)):
            firstNode = currentPath[i]
            secondNode = currentPath[j]

            action_flags[self.edgesDict[str(firstNode)+':'+str(secondNode)]] += 1.0
            i = i + 1
            j = j + 1
    
    def mark_action_to_edges(self, first_node, init_source, final_destination): 
       
        action_flags = np.zeros(self.numEdges)

        self.mark_edges(action_flags, init_source, first_node, init_source, final_destination)

        if first_node!=final_destination:
            self.mark_edges(action_flags, first_node, final_destination, init_source, final_destination)
        
        return action_flags

    # def compute_srnode(self):
    #     node_mll=dict()
    #     for i in self.graph:
    #         MLL = 0
    #         for j in self.graph[i]:
    #             # position=self.edgesDict[str(i)+':'+str(j)]
    #             if MLL < self.graph[i][j][0]['utilization']:
    #                 MLL = self.graph[i][j][0]['utilization']
    #         node_mll[str(i)] = MLL
    #     rate=int(np.ceil(self.numNodes*self.sr_rate))
    #     sorted_dict = list(sorted(node_mll.items(), key=lambda x:x[1], reverse= True))[:rate]
    #     for i in range(len(sorted_dict)):
    #         self.srnode.append(int(sorted_dict[i][0]))
        # print(self.srnode)


    def precompute_all_shortest_paths(self, G, weight='weight'):
        self.spath = {}
        for u in G.nodes():
            _, paths = nx.single_source_dijkstra(G, u, weight=weight)
            self.spath[u] = {v: paths.get(v, None) for v in G.nodes()}
        return self.spath

    def concat_paths(self, *paths):
        if not paths:
            return None
        result = []
        for p in paths:
            if not p:
                return None
            if not result:
                result.extend(p)
            else:
                if result[-1] == p[0]:
                    result.extend(p[1:])
                else:
                    result.extend(p)
        if len(result) != len(dict.fromkeys(result)):
            return None
        return result

    def build_path_from_spec(self,src, dst, insertion, segments, spath):
        if insertion is None:
            if segments:
                return None
            return spath[src].get(dst)
        prefix = spath[src].get(insertion)
        if not prefix:
            return None
        if len(segments) == 0:
            return spath[src].get(dst)
        elif len(segments) == 1:
            S1 = segments[0]
            mid = spath[insertion].get(S1)
            if not mid:
                return None
            suffix = spath[S1].get(dst)
            if not suffix:
                return None
            return self.concat_paths(prefix, mid, suffix)
        elif len(segments) == 2:
            S1, S2 = segments
            mid1 = spath[insertion].get(S1)
            if not mid1:
                return None
            mid2 = spath[S1].get(S2)
            if not mid2:
                return None
            suffix = spath[S2].get(dst)
            if not suffix:
                return None
            return self.concat_paths(prefix, mid1, mid2, suffix)
        else:
            return None

    def enumerate_candidate_paths(self, G, sr_nodes, src, dst, max_segments=2, weight='weight'):

        candidates = []

        p0 = self.spath[src].get(dst)
        if p0:
            candidates.append({
                "path": p0,
                "insertion": None,
                "segments": [],
                "type": "shortest",
                "spath": self.spath
            })

        for I in sr_nodes:
            if self.spath[src].get(I) is None:
                continue
            
            for S1 in sr_nodes:
                path1 = self.build_path_from_spec(src, dst, I, [S1], self.spath)
                if path1:
                    candidates.append({
                        "path": path1,
                        "insertion": I,
                        "segments": [S1],
                        "type": "1SR",
                        "spath": self.spath
                    })

            if max_segments >= 2:
                for S1, S2 in product(sr_nodes, repeat=2):
                    if S1 == S2:
                        continue
                    path2 = self.build_path_from_spec(src, dst, I, [S1, S2], self.spath)
                    if path2:
                        candidates.append({
                            "path": path2,
                            "insertion": I,
                            "segments": [S1, S2],
                            "type": "2SR",
                            "spath": self.spath
                        })

        priority = {"shortest": 0, "1SR": 1, "2SR": 2}
        unique = {}
        for c in candidates:
            key = tuple(c["path"])
            existing = unique.get(key)
            if existing is None or priority[c["type"]] > priority[existing["type"]]:
                unique[key] = c
        return list(unique.values())

    def refine_active_sr_nodes(self, candidate, src, dst):
        spath = candidate["spath"]
        path = candidate["path"]
        default = spath[src].get(dst)
        if default and tuple(default) == tuple(path):
            return []  

        original_nodes = []
        if candidate["insertion"] is not None:
            original_nodes.append(("insertion", candidate["insertion"]))
        for idx, s in enumerate(candidate["segments"]):
            original_nodes.append((f"seg{idx}", s))

        active = set()
        for kind, node in original_nodes:
            new_insertion = candidate["insertion"]
            new_segments = list(candidate["segments"])
            if kind == "insertion":
                new_insertion = None
            else:
                new_segments = [s for s in new_segments if s != node]
            new_path = self.build_path_from_spec(src, dst, new_insertion, new_segments, spath)
            if new_path is None or tuple(new_path) != tuple(path):
                active.add(node)
        return sorted(active)

    def path_cost(G, path, weight='weight'):
        cost = 0.0
        for u, v in zip(path, path[1:]):
            cost += G[u][v].get(weight, 1.0)
        return cost

    def compute_all_pairs_topk(self, G, sr_nodes, kp=4, max_segments=2, weight='weight'):
        nodes = list(G.nodes())
        self.src_dst_k_middlepoints = {} 
        for src in nodes:
            self.src_dst_k_middlepoints.setdefault(src, {})
            for dst in nodes:
                if src == dst:
                    continue
                candidates = self.enumerate_candidate_paths(G, sr_nodes, src, dst, max_segments=max_segments, weight=weight)
                enriched = []
                for c in candidates:
                    cost = self.path_cost(G, c["path"], weight=weight)
                    active_sr = self.refine_active_sr_nodes(c, src, dst)
                    enriched.append({
                        "path": c["path"],
                        "type": c["type"],
                        "insertion": c["insertion"],
                        "segments": c["segments"],
                        "active_sr": active_sr,
                        "cost": cost
                    })
                # 取前 K 条最短（按 cost, 再少用 SR ）
                topk = sorted(enriched, key=lambda x: (x["cost"], len(x["active_sr"])))[:kp]
                self.src_dst_k_middlepoints[src][dst] = topk


    def compute_SPs(self):
        diameter = nx.diameter(self.graph)
        self.shortest_paths = np.zeros((self.numNodes,self.numNodes),dtype=object)
        
        allPaths = dict()
        sp_path = self.dataset_folder_name+"/shortest_paths.json"

        if not os.path.isfile(sp_path):
            for n1 in range (0,self.numNodes):
                for n2 in range (0,self.numNodes):
                    if (n1 != n2):
                        allPaths[str(n1)+':'+str(n2)] = []

                        [allPaths[str(n1)+':'+str(n2)].append(p) for p in nx.all_simple_paths(self.graph, source=n1, target=n2, cutoff=diameter*2)]                    # We take all the paths from n1 to n2 and we order them according to the path length

                        aux_sorted_paths = sorted(allPaths[str(n1)+':'+str(n2)], key=lambda item: (len(item), item))                    # self.shortest_paths[n1,n2] = nx.shortest_path(self.graph, n1, n2,weight='weight')
                        allPaths[str(n1)+':'+str(n2)] = aux_sorted_paths[0]
        
            with open(sp_path, 'w') as fp:
                json.dump(allPaths, fp)
        else:
            allPaths = json.load(open(sp_path))

        for n1 in range (0,self.numNodes):
            for n2 in range (0,self.numNodes):
                if (n1 != n2):
                    self.shortest_paths[n1,n2] = allPaths[str(n1)+':'+str(n2)]
        # print(self.shortest_paths)
        
    def _first_second(self):

        first = list()
        second = list()

        for i in self.graph:
            for j in self.graph[i]:
                neighbour_edges = self.graph.edges(j)

                for m, n in neighbour_edges:
                    if ((i != m or j != n) and (i != n or j != m)):
                        first.append(self.edgesDict[str(i) +':'+ str(j)])
                        second.append(self.edgesDict[str(m) +':'+ str(n)])

        self.first = tf.convert_to_tensor(first, dtype=tf.int32)
        self.second = tf.convert_to_tensor(second, dtype=tf.int32)

    def generate_environment(self, dataset_folder_name, graph_topology_name, EPISODE_LENGTH, K, X, rate):
        self.episode_length = EPISODE_LENGTH
        self.graph_topology_name = graph_topology_name
        self.dataset_folder_name = dataset_folder_name
        self.list_eligible_demands = list()
        self.iter_list_elig_demn = 0
        self.percentage_demands = X
        self.sr_rate = rate

        self.maxCapacity = 0


        graph_file = self.dataset_folder_name+"/"+self.graph_topology_name+".graph"

        results_file = self.dataset_folder_name+"/res_"+self.graph_topology_name+"_0"
        tm_file = self.dataset_folder_name+"/TM/"+self.graph_topology_name+".0.demands"
        self.defoDatasetAPI = defoResults.Defo_results(graph_file,results_file)

        self.graph = self.defoDatasetAPI.Gbase
        self.add_features_to_edges()
        self.numNodes = len(self.graph.nodes())
        self.numEdges = len(self.graph.edges())
        btwns = nx.edge_betweenness_centrality(self.graph)
        
        self.K = K
        if self.K>self.numNodes:
            self.K = self.numNodes

        self.edge_state = np.zeros((self.numEdges, 4))

        self.sredge_state= np.zeros((self.numEdges, 1))
        self.betweenness_centrality = np.zeros(self.numEdges) 
        self.shortest_paths = np.zeros((self.numNodes,self.numNodes),dtype="object")

        position = 0
        for i in self.graph:
            for j in self.graph[i]:
                self.edgesDict[str(i)+':'+str(j)] = position
                self.graph[i][j][0]['capacity'] = self.defoDatasetAPI.links_bw[i][j]
                self.graph[i][j][0]['weight'] = self.defoDatasetAPI.links_weight[i][j]
                if self.graph[i][j][0]['capacity']>self.maxCapacity:
                    self.maxCapacity = self.graph[i][j][0]['capacity']
                self.edge_state[position][1] = self.graph[i][j][0]['capacity']
                self.betweenness_centrality[position] = btwns[i,j]
                self.graph[i][j][0]['utilization'] = 0.0
                self.graph[i][j][0]['crossing_paths'].clear()
                position += 1

        self._first_second()
        self.firstTrueSize = len(self.first)

        self.link_capacity_feature = tf.convert_to_tensor(np.divide(self.edge_state[:,1], self.maxCapacity), dtype=tf.float32)
        self.betweenness_centrality = tf.convert_to_tensor(self.betweenness_centrality, dtype=tf.float32)


        self.nodes = list(range(0,self.numNodes))
        

        

    def step(self, action, demand, source, destination):

        self.episode_over = False
        self.reward = 0


        middlePointList = self.src_dst_k_middlepoints[source][destination]
        middlePoint = middlePointList[action]

        self.allocate_to_destination_sp_init(source, middlePoint, source, destination)

        if len(middlePoint['active_sr']) > 0:


            self.sp_middlepoints[str(source)+':'+str(destination)] = middlePoint
        
        self.sp_middlepoints_step = self.sp_middlepoints
        
        old_Utilization = self.edgeMaxUti[2]
        self.edgeMaxUti = (0, 0, 0)
        self.edgeMinUti = (0, 0, 0)
        for i in self.graph:
            for j in self.graph[i]:
                position = self.edgesDict[str(i)+':'+str(j)]
                self.edge_state[position][0] = self.graph[i][j][0]['utilization']
                link_capacity = self.links_bw[i][j]
                norm_edge_state_capacity = self.edge_state[position][0]/link_capacity
                if norm_edge_state_capacity>self.edgeMaxUti[2]:
                    self.edgeMaxUti = (i, j, norm_edge_state_capacity)
        self.currentVal = -self.edgeMaxUti[2]

        self.reward = np.around((old_Utilization-self.edgeMaxUti[2])*10,2)


        if self.iter_list_elig_demn<len(self.list_eligible_demands):
            self._obtain_demand()
        else:
            src = 1
            dst = 2
            bw = self.TM[src][dst]
            self.patMaxBandwth = (src, dst, int(bw))
            self.episode_over = True


        if str(self.patMaxBandwth[0])+':'+str(self.patMaxBandwth[1]) in self.sp_middlepoints:

            middlepoint = self.sp_middlepoints[str(self.patMaxBandwth[0])+':'+str(self.patMaxBandwth[1])]

            self.decrease_links_utilization_sp_init(self.patMaxBandwth[0], middlepoint, self.patMaxBandwth[0], self.patMaxBandwth[1])
            
            del self.sp_middlepoints[str(self.patMaxBandwth[0])+':'+str(self.patMaxBandwth[1])] 
        else: 
            self.decrease_links_utilization_sp(self.patMaxBandwth[0], self.patMaxBandwth[1], self.patMaxBandwth[0], self.patMaxBandwth[1])
        

        self.edge_state[:,2] = 0
        self.edge_state[:,3] = 0

        return self.reward, self.episode_over, 0.0, self.TM[self.patMaxBandwth[0]][self.patMaxBandwth[1]], self.patMaxBandwth[0], self.patMaxBandwth[1], self.edgeMaxUti, 0.0, np.std(self.edge_state[:,0])



    def compute_ospf_srnode(self):
        node_mll=dict()
        for i in self.graph:
            MLL = 0
            for j in self.graph[i]:
                # position=self.edgesDict[str(i)+':'+str(j)]
                if MLL < self.graph[i][j][0]['utilization']:
                    MLL = self.graph[i][j][0]['utilization']
            node_mll[str(i)] = MLL
        rate=int(np.ceil(self.numNodes*self.sr_rate))

        sorted_dict = list(sorted(node_mll.items(), key=lambda x:x[1], reverse= True))[:rate]

        self.ospfnode=[]
        for i in range(len(sorted_dict)):
            self.ospfnode.append(int(sorted_dict[i][0]))


    def reset(self, tm_id, choice, sr_node):

        self.srnode = []

        self._generate_tm(tm_id)
        self.sp_middlepoints = dict()
        self.compute_SPs()
        self.precompute_all_shortest_paths(self.graph)

  
        self.compute_link_utilization_reset()
        self.compute_ospf_srnode()

        if choice == 1:
            self.srnode = sr_node
        elif choice == 3:
            self.srnode = list(self.graph.nodes)

    
        self.compute_all_pairs_topk(self.graph, self.srnode, 4, max_segments=2)
        
   
        self.edgeMaxUti = (0, 0, 0)

        list_link_uti_id = list()
        for i in self.graph:
            for j in self.graph[i]:
                position = self.edgesDict[str(i)+':'+str(j)]
                self.edge_state[position][0] = self.graph[i][j][0]['utilization']
                self.edge_state[position][1] = self.graph[i][j][0]['capacity']
                link_capacity = self.links_bw[i][j]

                list_link_uti_id.append((self.edge_state[position][0], i, j))

                norm_edge_state_capacity = self.edge_state[position][0]/link_capacity
                if norm_edge_state_capacity>self.edgeMaxUti[2]:
                    self.edgeMaxUti = (i, j, norm_edge_state_capacity)

        if self.top_K_critical_demands:

            list_link_uti_id = sorted(list_link_uti_id, key=lambda tup: tup[0], reverse=True)[:self.num_critical_links]
            self._get_top_k_critical_flows(list_link_uti_id)

        self.currentVal = -self.edgeMaxUti[2]
        self.initial_maxLinkUti = -self.edgeMaxUti[2]

        self._obtain_demand()

   
        self.decrease_links_utilization_sp(self.patMaxBandwth[0], self.patMaxBandwth[1], self.patMaxBandwth[0], self.patMaxBandwth[1])

      
        self.edge_state[:,2] = 0
        self.edge_state[:,3] = 0

        return self.TM[self.patMaxBandwth[0]][self.patMaxBandwth[1]], self.patMaxBandwth[0], self.patMaxBandwth[1]


    def get_srnode(self, rate):
        num = int(np.ceil(self.numNodes * rate))
        for n in self.graph:
            self.flowAmountDict[str(n)] = 0

        for k,v in self.sp_middlepoints_step.items():
            init_source = k.split(':')[0]
            final_destination = k.split(':')[1]
            srnode = v['active_sr']

            oribw = self.ori_tm[init_source][final_destination]
            srhlen = (8+16*(len(srnode)-1) +40)*8/1000
            add_bw = srhlen * (self.tm0rt + self.tm1rt + self.tm2rt)
            new_bw = add_bw+oribw
            for n in srnode:
                self.flowAmountDict[n] += new_bw

        sortdict = sorted(self.flowAmountDict, key = self.flowAmountDict.get, reverse=True)[:num]
        sr_node=[]
        for i in sortdict:
            sr_node.append(int(i))

        return sr_node
        
            


    def allocate_to_destination_sp(self, src, dst, init_source, final_destination): 
        bw_allocate = self.TM[init_source][final_destination]
        currentPath = self.shortest_paths[src,dst]


        
        i = 0
        j = 1
        num = 0
        # try:
        while (j < len(currentPath)):
            firstNode = currentPath[i]
            secondNode = currentPath[j]

            self.graph[firstNode][secondNode][0]['utilization'] += bw_allocate  
            self.graph[firstNode][secondNode][0]['crossing_paths'][str(init_source)+':'+str(final_destination)] = bw_allocate
            self.edge_state[self.edgesDict[str(firstNode)+':'+str(secondNode)]][0] = self.graph[firstNode][secondNode][0]['utilization']
            i = i + 1
            j = j + 1




    # def get_additional_bandConsum(self):

    #     add_band = 0

    #     num = 0
    #     totalflowexpand = 0
    #     for src in range (0,self.numNodes):
    #         for dst in range (0,self.numNodes):
    #             if src != dst:
    #                 key = str(src)+":"+str(dst)
    #                 if key in self.sp_middlepoints:
    #                     mid = self.sp_middlepoints[key]

    #                     if mid != dst:

    #                         arrival = random.uniform(60,200)
    #                         self.srbandConsume[key] = 0.32 * arrival 

    #     for k, v in self.srbandConsume.items():
    #         add_band += v
    #         num += 1

    #         src = int(k.split(':')[0])
    #         dst = int(k.split(':')[1])
    #         mid = self.sp_middlepoints[k]
    #         p1 = self.shortest_paths[src,mid]
    #         i=0
    #         j=1
    #         while(j<len(p1)):
    #             n1 = p1[i]
    #             n2 = p1[j]
    #             self.sredge_state[self.edgesDict[str(n1)+':'+str(n2)]][0] += v
    #             i += 1
    #             j += 1
            
    #         p2 = self.shortest_paths[mid,dst]
    #         self.srpathlength[k] = len(p1) +len(p2) - 1
    #         i=0
    #         j=1
    #         while(j<len(p2)):
    #             n1 = p2[i]
    #             n2 = p2[j]
    #             self.sredge_state[self.edgesDict[str(n1)+':'+str(n2)]][0] += v
    #             i += 1
    #             j += 1

    #         flowexpand = v/self.TM[src][dst]
    #         totalflowexpand += (1+flowexpand)

    #     avaflowexpand = totalflowexpand /num

    #     link_band = 0

    #     for a in self.sredge_state[:,0]:
    #         link_band += a

    #     ava_link_band = link_band / self.numEdges

    #     avalinkratio = ava_link_band / self.maxCapacity

    #     if num==0:
    #         ava_flowband=0
    #     else:

    #         ava_flowband = add_band / num


    #     totalpathlen = 0
    #     totalextendlen = 0
    #     for src in range (0,self.numNodes):
    #         for dst in range (0,self.numNodes):
    #             if src != dst:
    #                 key = str(src)+':'+str(dst)
    #                 if key not in self.srpathlength:
    #                     totalpathlen += len(self.shortest_paths[src,dst])
    #                 else:
    #                     originpathlen = len(self.shortest_paths[src,dst])
    #                     totalpathlen += self.srpathlength[key]
    #                     totalextendlen += self.srpathlength[key] / originpathlen

    #     avapathlen = totalpathlen /  (self.numNodes * (self.numNodes-1))
    #     if num==0:

    #         avaextendlen=0
    #     else:
    #         avaextendlen = totalextendlen / num


    #     return num, link_band, ava_link_band, avalinkratio, ava_flowband, avaflowexpand, avapathlen, avaextendlen
        


    def allocate_to_destination_sp_init(self, src, pdict, init_source, final_destination): 

        path = pdict['path']

        srnode = pdict['active_sr']

        if len(srnode) == 0:
            bw_allocate = self.TM[init_source][final_destination]
            while (j < len(path)):
                firstNode = path[i]
                secondNode = path[j]
                

                self.graph[firstNode][secondNode][0]['utilization'] += bw_allocate  
                self.graph[firstNode][secondNode][0]['crossing_paths'][str(init_source)+':'+str(final_destination)] = bw_allocate
                self.edge_state[self.edgesDict[str(firstNode)+':'+str(secondNode)]][0] = self.graph[firstNode][secondNode][0]['utilization']
                i = i + 1
                j = j + 1
        
        if len(srnode)==1:
            oribw = self.ori_tm[init_source][final_destination]
            srnode0 = srnode[0]
            srnode1 = srnode[1]
            srhlen = (8+16+40)*8/1000
            add_bw = srhlen * (self.tm0rt + self.tm1rt + self.tm2rt)
            new_bw = add_bw+oribw
            flag = 0
            while (j < len(path)):
                firstNode = path[i]
                secondNode = path[j]
                if firstNode != srnode0 and flag != 1:
                    
                    self.graph[firstNode][secondNode][0]['utilization'] += oribw  
                    self.graph[firstNode][secondNode][0]['crossing_paths'][str(init_source)+':'+str(final_destination)] = oribw
                    self.edge_state[self.edgesDict[str(firstNode)+':'+str(secondNode)]][0] = self.graph[firstNode][secondNode][0]['utilization']
                    i = i + 1
                    j = j + 1
                if secondNode == srnode1:
                    flag = 0
                    self.graph[firstNode][secondNode][0]['utilization'] += new_bw  
                    self.graph[firstNode][secondNode][0]['crossing_paths'][str(init_source)+':'+str(final_destination)] = new_bw
                    self.edge_state[self.edgesDict[str(firstNode)+':'+str(secondNode)]][0] = self.graph[firstNode][secondNode][0]['utilization']
                    i = i + 1
                    j = j + 1
                else:
                    flag = 1
                    self.graph[firstNode][secondNode][0]['utilization'] += new_bw  
                    self.graph[firstNode][secondNode][0]['crossing_paths'][str(init_source)+':'+str(final_destination)] = new_bw
                    self.edge_state[self.edgesDict[str(firstNode)+':'+str(secondNode)]][0] = self.graph[firstNode][secondNode][0]['utilization']
                    i = i + 1
                    j = j + 1

        if len(srnode)==2:
            oribw = self.ori_tm[init_source][final_destination]
            srnode0 = srnode[0]
            srnode1 = srnode[1]
            srnode2 = srnode[2]
            srhlen = (8+32+40)*8/1000
            add_bw = srhlen * (self.tm0rt + self.tm1rt + self.tm2rt)
            new_bw = add_bw+oribw
            flag = 0
            while (j < len(path)):

                firstNode = path[i]
                secondNode = path[j]
                if firstNode != srnode0 and flag != 1:
                    
                    self.graph[firstNode][secondNode][0]['utilization'] += oribw  
                    self.graph[firstNode][secondNode][0]['crossing_paths'][str(init_source)+':'+str(final_destination)] = oribw
                    self.edge_state[self.edgesDict[str(firstNode)+':'+str(secondNode)]][0] = self.graph[firstNode][secondNode][0]['utilization']
                    i = i + 1
                    j = j + 1
                if secondNode == srnode2:
                    flag = 0
                    self.graph[firstNode][secondNode][0]['utilization'] += new_bw  
                    self.graph[firstNode][secondNode][0]['crossing_paths'][str(init_source)+':'+str(final_destination)] = new_bw
                    self.edge_state[self.edgesDict[str(firstNode)+':'+str(secondNode)]][0] = self.graph[firstNode][secondNode][0]['utilization']
                    i = i + 1
                    j = j + 1
                else:
                    flag = 1
                    self.graph[firstNode][secondNode][0]['utilization'] += new_bw  
                    self.graph[firstNode][secondNode][0]['crossing_paths'][str(init_source)+':'+str(final_destination)] = new_bw
                    self.edge_state[self.edgesDict[str(firstNode)+':'+str(secondNode)]][0] = self.graph[firstNode][secondNode][0]['utilization']
                    i = i + 1
                    j = j + 1
        
 
    def mark_action_sp(self, src, dst, init_source, final_destination): 
        bw_allocate = self.TM[init_source][final_destination]
        currentPath = self.shortest_paths[src,dst]
        
        i = 0
        j = 1
        

        while (j < len(currentPath)):
            firstNode = currentPath[i]
            secondNode = currentPath[j]

            self.edge_state[self.edgesDict[str(firstNode)+':'+str(secondNode)]][2] = bw_allocate/self.edge_state[self.edgesDict[str(firstNode)+':'+str(secondNode)]][1]
            i = i + 1
            j = j + 1
    
    def mark_action_sp_init(self, src, pdict, init_source, final_destination): 

        path = pdict['path']

        srnode = pdict['active_sr']

        if len(srnode) == 0:
            bw_allocate = self.TM[init_source][final_destination]
            while (j < len(path)):
                firstNode = path[i]
                secondNode = path[j]
                
                self.edge_state[self.edgesDict[str(firstNode)+':'+str(secondNode)]][2] = bw_allocate/self.edge_state[self.edgesDict[str(firstNode)+':'+str(secondNode)]][1]
                i = i + 1
                j = j + 1
        
        if len(srnode)==1:
            oribw = self.ori_tm[init_source][final_destination]
            srnode0 = srnode[0]
            srnode1 = srnode[1]
            srhlen = (8+16+40)*8/1000
            add_bw = srhlen * (self.tm0rt + self.tm1rt + self.tm2rt)
            new_bw = add_bw+oribw
            flag = 0
            while (j < len(path)):
                firstNode = path[i]
                secondNode = path[j]
                if firstNode != srnode0 and flag != 1:
                    self.edge_state[self.edgesDict[str(firstNode)+':'+str(secondNode)]][2] = oribw/self.edge_state[self.edgesDict[str(firstNode)+':'+str(secondNode)]][1]
                    # self.edge_state[self.edgesDict[str(firstNode)+':'+str(secondNode)]][3] = add_bw/self.edge_state[self.edgesDict[str(firstNode)+':'+str(secondNode)]][1]
                    i = i + 1
                    j = j + 1
                if secondNode == srnode1:
                    flag = 0
                    self.edge_state[self.edgesDict[str(firstNode)+':'+str(secondNode)]][2] = new_bw/self.edge_state[self.edgesDict[str(firstNode)+':'+str(secondNode)]][1]
                    self.edge_state[self.edgesDict[str(firstNode)+':'+str(secondNode)]][3] = add_bw/self.edge_state[self.edgesDict[str(firstNode)+':'+str(secondNode)]][1]
                    i = i + 1
                    j = j + 1
                else:
                    flag = 1
                    self.edge_state[self.edgesDict[str(firstNode)+':'+str(secondNode)]][2] = new_bw/self.edge_state[self.edgesDict[str(firstNode)+':'+str(secondNode)]][1]
                    self.edge_state[self.edgesDict[str(firstNode)+':'+str(secondNode)]][3] = add_bw/self.edge_state[self.edgesDict[str(firstNode)+':'+str(secondNode)]][1]
                    i = i + 1
                    j = j + 1

        if len(srnode)==2:
            oribw = self.ori_tm[init_source][final_destination]
            srnode0 = srnode[0]
            srnode1 = srnode[1]
            srnode2 = srnode[2]
            srhlen = (8+32+40)*8/1000
            add_bw = srhlen * (self.tm0rt + self.tm1rt + self.tm2rt)
            new_bw = add_bw+oribw
            flag = 0
            while (j < len(path)):
                firstNode = path[i]
                secondNode = path[j]
                if firstNode != srnode0 and flag != 1:
                    self.edge_state[self.edgesDict[str(firstNode)+':'+str(secondNode)]][2] = oribw/self.edge_state[self.edgesDict[str(firstNode)+':'+str(secondNode)]][1]
                    # self.edge_state[self.edgesDict[str(firstNode)+':'+str(secondNode)]][3] = add_bw/self.edge_state[self.edgesDict[str(firstNode)+':'+str(secondNode)]][1]
                    i = i + 1
                    j = j + 1
                if secondNode == srnode2:
                    flag = 0
                    self.edge_state[self.edgesDict[str(firstNode)+':'+str(secondNode)]][2] = new_bw/self.edge_state[self.edgesDict[str(firstNode)+':'+str(secondNode)]][1]
                    self.edge_state[self.edgesDict[str(firstNode)+':'+str(secondNode)]][3] = add_bw/self.edge_state[self.edgesDict[str(firstNode)+':'+str(secondNode)]][1]
                    i = i + 1
                    j = j + 1
                else:
                    flag = 1
                    self.edge_state[self.edgesDict[str(firstNode)+':'+str(secondNode)]][2] = new_bw/self.edge_state[self.edgesDict[str(firstNode)+':'+str(secondNode)]][1]
                    self.edge_state[self.edgesDict[str(firstNode)+':'+str(secondNode)]][3] = add_bw/self.edge_state[self.edgesDict[str(firstNode)+':'+str(secondNode)]][1]
                    i = i + 1
                    j = j + 1
