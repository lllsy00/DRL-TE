import gym
import numpy as np
import networkx as nx
import random
from gym import error, spaces, utils
from random import choice
import random
import pandas as pd
import pickle
import json 
import os.path
import gc
import time
import defo_process_results as defoResults
from itertools import product


class Env15(gym.Env):

    def __init__(self):
        self.arrval_dict = {}
        self.packet_len = 1
        self.afterAllocateBandDict = {}

        self.graph = None 
        self.source = None
        self.destination = None
        self.demand = None

        self.edge_state = None
        self.graph_topology_name = None 
        self.dataset_folder_name = None 

        self.diameter = None
        self.list_of_demands_to_change = None       
        self.between_feature = None

        self.sp_middlepoints = None 
        self.shortest_paths = None 
        
        self.srbandConsume = dict()
        
        self.srpathlength = dict()
        
        self.mu_bet = None
        self.std_bet = None
        self.episode_length = None

        self.list_eligible_demands = None 
        self.num_critical_links = 5

        self.error_evaluation = None

        self.target_link_capacity = None

        self.TM = None 
        self.meanTM = None
        self.stdTM = None
        self.sumTM = None
        self.routing = None 
        self.paths_Matrix_from_routing = None 

        self.K = None
        self.nodes = None 
        self.ordered_edges = None
        self.edgesDict = dict() 
        self.previous_path = None

        self.src_dst_k_middlepoints = None
        self.node_to_index_dic = None
        self.index_to_node_lst = None 

        self.numNodes = None
        self.numEdges = None
        self.numSteps = 0 

        self.sameLink = False 

        self.edgeMaxUti = None 

        self.patMaxBandwth = None 
        self.maxBandwidth = None

        self.episode_over = True
        self.reward = 0
        self.allPaths = dict() 
        self.srnode= list()
        self.sr_rate= None
        self.traffic = list()
        self.splist=dict()

    def seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
    
    def add_features_to_edges(self):
        incId = 1
        for node in self.graph:
            for adj in self.graph[node]:
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
                if not 'crossing_paths' in self.graph[node][adj][0]: 
                    self.graph[node][adj][0]['crossing_paths'] = dict()
                incId = incId + 1
    
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

    def _generate_tm(self, tm_id):        

        graph_file = self.dataset_folder_name+"/"+self.graph_topology_name+".graph"
        results_file = self.dataset_folder_name+"/res_"+self.graph_topology_name+"_"+str(tm_id)
        tm_file = self.dataset_folder_name+"/TM/"+self.graph_topology_name+'.'+str(tm_id)+".demands"
        
        self.defoDatasetAPI = defoResults.Defo_results(graph_file,results_file)
        self.links_bw = self.defoDatasetAPI.links_bw
        graph_file = self.dataset_folder_name+"/"+self.graph_topology_name+".graph"

        results_file = self.dataset_folder_name+"/res_"+self.graph_topology_name+"_"+str(tm_id)
        tm_file = self.dataset_folder_name+"/TM/"+self.graph_topology_name+'.'+str(tm_id)+".demands"
        
        self.defoDatasetAPI = defoResults.Defo_results(graph_file,results_file)
        # self.links_bw = self.defoDatasetAPI.links_bw
        self.MP_matrix = self.defoDatasetAPI.MP_matrix
        self.ori_tm, self.tm0bw, self.tm0ps, self.tm0rt, self.tm1bw, self.tm1ps, self.tm1rt, self.tm2bw, self.tm2ps, self.tm2rt = self.defoDatasetAPI._get_traffic_matrix(tm_file)
        self.TM=self.ori_tm

        self.maxBandwidth = np.amax(self.TM)

        traffic = np.copy(self.TM)

        traffic = traffic[~np.eye(traffic.shape[0], dtype=bool)].reshape(traffic.shape[0], -1)

        self.sumTM = np.sum(traffic)
        self.target_link_capacity = self.sumTM/self.numEdges
        self.meanTM = np.mean(traffic)
        self.stdTM = np.std(traffic)
    
    def compute_link_utilization_reset_sp(self):  
        for src in range (0,self.numNodes):
            for dst in range (0,self.numNodes):
                if src!=dst:
                    self.allocate_to_destination_sp(src, dst, src, dst)

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
    #             if MLL < self.graph[i][j][0]['utilization']:
    #                 MLL = self.graph[i][j][0]['utilization']
    #         node_mll[str(i)] = MLL
    #     rate=int(np.ceil(self.numNodes*self.sr_rate))
    #     sorted_dict = list(sorted(node_mll.items(), key=lambda x:x[1], reverse= True))[:rate]
    #     for i in range(len(sorted_dict)):
    #         self.srnode.append(int(sorted_dict[i][0]))                          

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
                        [allPaths[str(n1)+':'+str(n2)].append(p) for p in nx.all_simple_paths(self.graph, source=n1, target=n2, cutoff=diameter*2)]
                        aux_sorted_paths = sorted(allPaths[str(n1)+':'+str(n2)], key=lambda item: (len(item), item))
                        allPaths[str(n1)+':'+str(n2)] = aux_sorted_paths[0]
        
            with open(sp_path, 'w') as fp:
                json.dump(allPaths, fp)
        else:
            allPaths = json.load(open(sp_path))
        for n1 in range (0,self.numNodes):
            for n2 in range (0,self.numNodes):
                if (n1 != n2):
                    self.shortest_paths[n1,n2] = allPaths[str(n1)+':'+str(n2)]

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

        
    def generate_environment(self, dataset_folder_name, graph_topology_name, EPISODE_LENGTH, K, percentage_demands,rate):  # 生成环境，计算中间节点
        self.episode_length = EPISODE_LENGTH
        self.graph_topology_name = graph_topology_name
        self.dataset_folder_name = dataset_folder_name
        self.list_eligible_demands = list()
        self.percentage_demands = percentage_demands
        self.sr_rate=rate

        self.maxCapacity = 0

        graph_file = self.dataset_folder_name+"/"+self.graph_topology_name+".graph"
        results_file = self.dataset_folder_name+"/res_"+self.graph_topology_name+"_0"
        tm_file = self.dataset_folder_name+"/TM/"+self.graph_topology_name+".0.demands"
        self.defoDatasetAPI = defoResults.Defo_results(graph_file,results_file)

        self.node_to_index_dic = self.defoDatasetAPI.node_to_index_dic_pvt
        self.index_to_node_lst = self.defoDatasetAPI.index_to_node_lst_pvt

        self.graph = self.defoDatasetAPI.Gbase
        self.add_features_to_edges()
        self.numNodes = len(self.graph.nodes())
        self.numEdges = len(self.graph.edges())

        self.K = K
        if self.K>self.numNodes:
            self.K = self.numNodes

        self.edge_state = np.zeros((self.numEdges, 2))
        self.sredge_state = np.zeros((self.numEdges,1))
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
                self.graph[i][j][0]['utilization'] = 0.0
                self.graph[i][j][0]['crossing_paths'].clear()
                position += 1

        self.nodes = list(range(0,self.numNodes))
        self.compute_SPs()

    
    def step_sp(self, action, source, destination):  
        middlePointList = list(self.src_dst_k_middlepoints[source][destination])
        middlePoint = middlePointList[action]

        self.allocate_to_destination_sp(source, middlePoint, source, destination)

        if middlePoint!=destination:
            self.allocate_to_destination_sp(middlePoint, destination, source, destination)
            self.sp_middlepoints[str(source)+':'+str(destination)] = middlePoint
        
        old_Utilization = self.edgeMaxUti[2]
        self.edgeMaxUti = (0, 0, 0)
        for i in self.graph:
            for j in self.graph[i]:
                position = self.edgesDict[str(i)+':'+str(j)]
                self.edge_state[position][0] = self.graph[i][j][0]['utilization']
                link_capacity = self.links_bw[i][j]
                norm_edge_state_capacity = self.edge_state[position][0]/link_capacity
                if norm_edge_state_capacity>self.edgeMaxUti[2]:
                    self.edgeMaxUti = (i, j, norm_edge_state_capacity)

        return self.edgeMaxUti[2]
    
    def step_hill_sp(self, action, source, destination):  
        middlePointList = list(self.src_dst_k_middlepoints[source][destination])
        middlePoint = middlePointList[action]
        self.allocate_to_destination_sp_init(source, middlePoint, source, destination)


        if len(middlePoint['active_sr']) > 0:


            self.sp_middlepoints[str(source)+':'+str(destination)] = middlePoint
        
        old_Utilization = self.edgeMaxUti[2]
        self.edgeMaxUti = (0, 0, 0)
        for i in self.graph:
            for j in self.graph[i]:
                position = self.edgesDict[str(i)+':'+str(j)]
                self.edge_state[position][0] = self.graph[i][j][0]['utilization']
                link_capacity = self.links_bw[i][j]
                norm_edge_state_capacity = self.edge_state[position][0]/link_capacity
                if norm_edge_state_capacity>self.edgeMaxUti[2]:
                    self.edgeMaxUti = (i, j, norm_edge_state_capacity)

        return -self.edgeMaxUti[2]

            

    
    def reset_sp(self, tm_id):
        self._generate_tm(tm_id)

        self.sp_middlepoints = dict()

        for i in self.graph:
            for j in self.graph[i]:
                self.graph[i][j][0]['utilization'] = 0.0
                self.graph[i][j][0]['crossing_paths'].clear()

        self.compute_link_utilization_reset_sp()
        for i in self.graph:
            for j in self.graph[i]:
                position = self.edgesDict[str(i)+':'+str(j)]
                self.edge_state[position][0] = self.graph[i][j][0]['utilization']
                self.edge_state[position][1] = self.graph[i][j][0]['capacity']
                link_capacity = self.links_bw[i][j]
                norm_edge_state_capacity = self.edge_state[position][0]/link_capacity
                if norm_edge_state_capacity>self.edgeMaxUti[2]:
                    self.edgeMaxUti = (i, j, norm_edge_state_capacity)
        
        return self.edgeMaxUti[2]

    def reset_hill_sp(self, tm_id, srnode, choice):

        self._generate_tm(tm_id)
        self.precompute_all_shortest_paths(self.graph)
        if choice==1:
            self.srnode = srnode
        self.sp_middlepoints = dict()

        for i in self.graph:
            for j in self.graph[i]:
                self.graph[i][j][0]['utilization'] = 0.0
                self.graph[i][j][0]['crossing_paths'].clear()

        self.compute_link_utilization_reset_sp()
        self.compute_all_pairs_topk(self.graph, self.srnode, 4, max_segments=2)

        self.edgeMaxUti = (0, 0, 0)
        for i in self.graph:
            for j in self.graph[i]:
                position = self.edgesDict[str(i)+':'+str(j)]
                self.edge_state[position][0] = self.graph[i][j][0]['utilization']
                self.edge_state[position][1] = self.graph[i][j][0]['capacity']
                link_capacity = self.links_bw[i][j]
                norm_edge_state_capacity = self.edge_state[position][0]/link_capacity
                if norm_edge_state_capacity>self.edgeMaxUti[2]:
                    self.edgeMaxUti = (i, j, norm_edge_state_capacity)
        
        return -self.edgeMaxUti[2]

    def _get_top_k_critical_flows(self, list_ids):
        self.list_eligible_demands.clear()
        for linkId in list_ids:
            i = linkId[0]
            j = linkId[1]
            for demand, value in self.graph[i][j][0]['crossing_paths'].items():
                src, dst = int(demand.split(':')[0]), int(demand.split(':')[1])
                if (src, dst, self.TM[src,dst]) not in self.list_eligible_demands:  
                    self.list_eligible_demands.append((src, dst, self.TM[src,dst]))

        self.list_eligible_demands = sorted(self.list_eligible_demands, key=lambda tup: tup[2], reverse=True)
        if len(self.list_eligible_demands)>int(np.ceil(self.numNodes*(self.numNodes-1)*self.percentage_demands)):
            self.list_eligible_demands = self.list_eligible_demands[:int(np.ceil(self.numNodes*(self.numNodes-1)*self.percentage_demands))]


    def reset_DRL_hill_sp(self, tm_id, best_routing, list_of_demands_to_change, srnode, choice):
        
        self.afterAllocateBandDict = {}
        if choice == 1:

            self.srnode = srnode

        self._generate_tm(tm_id)
        self.precompute_all_shortest_paths(self.graph)
        if best_routing is not None: 
            self.sp_middlepoints = best_routing
        else: 
            self.sp_middlepoints = dict()
        self.list_of_demands_to_change = list_of_demands_to_change

        for i in self.graph:
            for j in self.graph[i]:
                self.graph[i][j][0]['utilization'] = 0.0
                self.graph[i][j][0]['crossing_paths'].clear()

        self.compute_link_utilization_reset_sp()


        self.compute_all_pairs_topk(self.graph, self.srnode, 4, max_segments=2)

        for key, middlepoint in self.sp_middlepoints.items(): 
            source = int(key.split(':')[0])
            dest = int(key.split(':')[1])

            if len(middlepoint['active_sr'])>0:
                self.decrease_links_utilization_sp(source, dest, source, dest)
                self.allocate_to_destination_sp_init(source, middlepoint, source, dest)      

        self.to_del=[]
        for key, middlepoint in self.sp_middlepoints.items():
            need_sr = middlepoint['active_sr']
            for i in need_sr:
                if i not in self.srnode:         
                    self.to_del.append(key)

        for i in self.to_del:
            source = int(i.split(':')[0])
            dest = int(i.split(':')[1])
            middlepoint=self.sp_middlepoints[i]
            self.decrease_links_utilization_sp_init(source, middlepoint, source, dest)

            self.allocate_to_destination_sp(source,dest,source,dest)
            del self.sp_middlepoints[i]

        self.edgeMaxUti = (0, 0, 0)

        list_link_uti_id = list()
        for i in self.graph:
            for j in self.graph[i]:
                position = self.edgesDict[str(i)+':'+str(j)]
                self.edge_state[position][0] = self.graph[i][j][0]['utilization']
                self.edge_state[position][1] = self.graph[i][j][0]['capacity']
                link_capacity = self.links_bw[i][j]

                list_link_uti_id.append((i, j, self.edge_state[position][0]))

                norm_edge_state_capacity = self.edge_state[position][0]/link_capacity
                if norm_edge_state_capacity>self.edgeMaxUti[2]:
                    self.edgeMaxUti = (i, j, norm_edge_state_capacity)
        
        list_link_uti_id = sorted(list_link_uti_id, key=lambda tup: tup[2], reverse=True)[:self.num_critical_links]
        self._get_top_k_critical_flows(list_link_uti_id)

        return -self.edgeMaxUti[2]
 

    def re_get_top_k(self):
        list_link_uti_id = list()
        for i in self.graph:
            for j in self.graph[i]:
                position = self.edgesDict[str(i)+':'+str(j)]
                self.edge_state[position][0] = self.graph[i][j][0]['utilization']
                self.edge_state[position][1] = self.graph[i][j][0]['capacity']
                link_capacity = self.links_bw[i][j]
                list_link_uti_id.append((i, j, self.edge_state[position][0]))

                norm_edge_state_capacity = self.edge_state[position][0]/link_capacity
                if norm_edge_state_capacity>self.edgeMaxUti[2]:
                    self.edgeMaxUti = (i, j, norm_edge_state_capacity)
        
        list_link_uti_id = sorted(list_link_uti_id, key=lambda tup: tup[2], reverse=True)[:self.num_critical_links]
        self._get_top_k_critical_flows(list_link_uti_id)
    
    def allocate_to_destination_sp(self, src, dst, init_source, final_destination): 
        bw_allocate = self.TM[init_source][final_destination]
        currentPath = self.shortest_paths[src,dst]
        
        i = 0
        j = 1

        while (j < len(currentPath)):
            firstNode = currentPath[i]
            secondNode = currentPath[j]

            self.graph[firstNode][secondNode][0]['utilization'] += bw_allocate  
            self.graph[firstNode][secondNode][0]['crossing_paths'][str(init_source)+':'+str(final_destination)] = bw_allocate
            self.edge_state[self.edgesDict[str(firstNode)+':'+str(secondNode)]][0] = self.graph[firstNode][secondNode][0]['utilization']
            i = i + 1
            j = j + 1

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
