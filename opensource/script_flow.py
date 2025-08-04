import numpy as np
import gym
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import json 
import gc
import gym_graph
import networkx as nx
import random
import matplotlib.pyplot as plt
import argparse
import time as tt
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
import actorPPOmiddR as actor
import pandas as pd
from collections import Counter
import pickle
import sys
from scipy.stats import entropy
import warnings
warnings.filterwarnings("ignore")

sys.setrecursionlimit(2000)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


ENV_MIDDROUT_AGENT_SP = 'GraphEnv-v25'
ENV_SIMM_ANEAL_AGENT = 'GraphEnv-v15'
ENV_SAP_AGENT = 'GraphEnv-v20'
SEED = 1
NUM_ACTIONS = 200
percentage_demands = 15 
str_perctg_demands = str(percentage_demands)
percentage_demands /= 100
rate = 0.3

repretm=-1



os.environ['PYTHONHASHSEED']=str(SEED)
np.random.seed(SEED)
tf.random.set_seed(5)


EPISODE_LENGTH_MIDDROUT = 100


MAX_NUM_EDGES = 100


hparamsDRLSP = {
    'l2': 0.005,
    'dropout_rate': 0.1,
    'link_state_dim': 20,
    'readout_units': 20,
    'learning_rate': 0.0002,
    'T': 4,
}

hidden_init_actor = tf.keras.initializers.Orthogonal(gain=np.sqrt(2), seed=SEED)
kernel_init_actor = tf.keras.initializers.Orthogonal(gain=np.sqrt(0.01), seed=SEED)

def old_cummax(alist, extractor):
    with tf.name_scope('cummax'):
        maxes = [tf.reduce_max(extractor(v)) + 1 for v in alist]
        cummaxes = [tf.zeros_like(maxes[0])]
        for i in range(len(maxes) - 1):
            cummaxes.append(tf.math.add_n(maxes[0:i + 1]))
    return cummaxes

class PPOMIDDROUTING_SP:  # env16
    def __init__(self, env_training):
        self.listQValues = None
        self.softMaxQValues = None

        self.action = None
        self.softMaxQValues = None
        self.listQValues = None
        self.K = env_training.K

        self.utilization_feature = None
        self.bw_allocated_feature = None

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=hparamsDRLSP['learning_rate'], beta_1=0.9, epsilon=1e-05)
        self.actor = actor.myModel(hparamsDRLSP, hidden_init_actor, kernel_init_actor)
        self.actor.build()

    def pred_action_node_distrib_sp(self, env, source, destination):

        list_k_features = list()

        middlePointList = env.src_dst_k_middlepoints[source][destination]
        itMidd = 0
        
        while itMidd < len(middlePointList):
            env.mark_action_sp_init(source, middlePointList[itMidd], source, destination)


            features = self.get_graph_features(env, source, destination)
            list_k_features.append(features)


            env.edge_state[:,2] = 0
            env.edge_state[:,3] = 0

            itMidd = itMidd + 1


        vs = [v for v in list_k_features]

        graph_ids = [tf.fill([tf.shape(vs[it]['link_state'])[0]], it) for it in range(len(list_k_features))]
        first_offset = old_cummax(vs, lambda v: v['first'])
        second_offset = old_cummax(vs, lambda v: v['second'])

        tensor = ({
            'graph_id': tf.concat([v for v in graph_ids], axis=0),
            'link_state': tf.concat([v['link_state'] for v in vs], axis=0),
            'first': tf.concat([v['first'] + m for v, m in zip(vs, first_offset)], axis=0),
            'second': tf.concat([v['second'] + m for v, m in zip(vs, second_offset)], axis=0),
            'num_edges': tf.math.add_n([v['num_edges'] for v in vs]),
            }
        )

        r = self.actor(tensor['link_state'], tensor['graph_id'], tensor['first'], tensor['second'], 
            tensor['num_edges'], training=False)
        self.listQValues = tf.reshape(r, (1, len(r)))

        self.softMaxQValues = tf.nn.softmax(self.listQValues)


        return self.softMaxQValues.numpy()[0], tensor
    
    def get_graph_features(self, env, source, destination):

        self.bw_allocated_feature = env.edge_state[:,2]
        self.utilization_feature = env.edge_state[:,0]
        self.srband = env.edge_state[:,3]
        sample = {
            'num_edges': env.numEdges,  
            'length': env.firstTrueSize,
            'capacity': env.link_capacity_feature,
            'bw_allocated': tf.convert_to_tensor(value=self.bw_allocated_feature, dtype=tf.float32),
            'utilization': tf.convert_to_tensor(value=np.divide(self.utilization_feature, env.edge_state[:,1]), dtype=tf.float32),
            'srband': tf.convert_to_tensor(value=self.srband, dtype=tf.float32),
            'first': env.first,
            'second': env.second
        }

        sample['utilization'] = tf.reshape(sample['utilization'][0:sample['num_edges']], [sample['num_edges'], 1])
        sample['capacity'] = tf.reshape(sample['capacity'][0:sample['num_edges']], [sample['num_edges'], 1])
        sample['bw_allocated'] = tf.reshape(sample['bw_allocated'][0:sample['num_edges']], [sample['num_edges'], 1])
        sample['srband'] = tf.reshape(sample['srband'][0:sample['num_edges']], [sample['num_edges'], 1])

        hiddenStates = tf.concat([sample['utilization'], sample['capacity'], sample['bw_allocated'], sample['srband']], axis=1)

        paddings = tf.constant([[0, 0], [0, hparamsDRLSP['link_state_dim'] - 3]])
        link_state = tf.pad(tensor=hiddenStates, paddings=paddings, mode="CONSTANT")

        inputs = {'link_state': link_state, 'first': sample['first'][0:sample['length']],
                'second': sample['second'][0:sample['length']], 'num_edges': sample['num_edges']}

        return inputs

def play_middRout_games_sp(tm_id, env_middRout_sp, agent, choice, sr_node):  
    demand, source, destination = env_middRout_sp.reset(tm_id, choice, sr_node)
    rewardAddTest = 0

    initMaxUti = env_middRout_sp.edgeMaxUti[2]
    OSPF_init = initMaxUti
    best_routing = env_middRout_sp.sp_middlepoints_step.copy()   
    srnode = env_middRout_sp.srnode
    list_of_demands_to_change = env_middRout_sp.list_eligible_demands


    start = tt.time()
    time_start_DRL = start
    while 1:   
        action_dist, tensor = agent.pred_action_node_distrib_sp(env_middRout_sp, source, destination)  
        action = np.argmax(action_dist) 
        
        reward, done, error_eval_links, demand, source, destination, maxLinkUti, minLinkUti, utiStd = env_middRout_sp.step(action, demand, source, destination)  
        rewardAddTest += reward
        if maxLinkUti[2]<initMaxUti:  
            initMaxUti = maxLinkUti[2]
            best_routing = env_middRout_sp.sp_middlepoints_step.copy()
            
        if done:
            break

    end = tt.time()
    
    return initMaxUti, end-start, OSPF_init, best_routing, list_of_demands_to_change, time_start_DRL, srnode

def play_help_games_sp(tm_id, env_middRout_sp, agent, choice, rate):  
    demand, source, destination = env_middRout_sp.reset(tm_id, choice, 1)
    rewardAddTest = 0

    initMaxUti = env_middRout_sp.edgeMaxUti[2]

    start = tt.time()
    time_start_DRL = start
    while 1:   
        action_dist, tensor = agent.pred_action_node_distrib_sp(env_middRout_sp, source, destination)  
        action = np.argmax(action_dist)  
        
        reward, done, error_eval_links, demand, source, destination, maxLinkUti, minLinkUti, utiStd = env_middRout_sp.step(action, demand, source, destination)  
        rewardAddTest += reward
        if maxLinkUti[2]<initMaxUti: 
            initMaxUti = maxLinkUti[2]

        if done:

            srnode = env_middRout_sp.get_srnode(rate)
            break
    end = tt.time()


    return srnode,env_middRout_sp.ospfnode

def play_help_drl_mll_sp(tm_id, env_middRout_sp, agent, choice, rate): 
    demand, source, destination = env_middRout_sp.reset(tm_id, choice, 1)
    rewardAddTest = 0

    initMaxUti = env_middRout_sp.edgeMaxUti[2]
    
    start = tt.time()
    time_start_DRL = start
    while 1:   
        action_dist, tensor = agent.pred_action_node_distrib_sp(env_middRout_sp, source, destination)  
        action = np.argmax(action_dist) 
        
        reward, done, error_eval_links, demand, source, destination, maxLinkUti, minLinkUti, utiStd = env_middRout_sp.step(action, demand, source, destination)  # 在环境中执行
        rewardAddTest += reward
        if maxLinkUti[2]<initMaxUti:  
            initMaxUti = maxLinkUti[2]

        if done:
            
            env_middRout_sp.compute_ospf_srnode()
            srnode = env_middRout_sp.ospfnode

            break
    end = tt.time()
    return srnode




class HILL_CLIMBING: 
    def __init__(self, env):
        self.num_actions = env.K 

    def get_value_sp(self, env, source, destination, action): 
        middlePointList = list(env.src_dst_k_middlepoints[source][destination])
        middlePoint = middlePointList[action]

        env.allocate_to_destination_sp_init(source, middlePoint, source, destination)


        env.sp_middlepoints[str(source)+':'+str(destination)] = middlePoint
        
        currentValue = -1000000  
        position = 0

        for i in env.graph:
            for j in env.graph[i]:
                link_capacity = env.links_bw[i][j]
                if env.edge_state[position][0]/link_capacity>currentValue:
                    currentValue = env.edge_state[position][0]/link_capacity
                position = position + 1
        

        if str(source)+':'+str(destination) in env.sp_middlepoints:
            middlepoint = env.sp_middlepoints[str(source)+':'+str(destination)]
            env.decrease_links_utilization_sp_init(source, middlepoint, source, destination)

            del env.sp_middlepoints[str(source)+':'+str(destination)] 
        else: 
            env.decrease_links_utilization_sp(source, destination, source, destination)
        
        return -currentValue
    
    def explore_neighbourhood_sp(self, env):
        dem_iter = 0
        nextVal = -1000000
        next_state = None

        for source in range(env.numNodes):
            for dest in range(env.numNodes):
                if source!=dest:
                    for action in range(len(env.src_dst_k_middlepoints[source][dest])):
                        middlepoint = -1
                        
                        if str(source)+':'+str(dest) in env.sp_middlepoints:
                            middlepoint = env.sp_middlepoints[str(source)+':'+str(dest)] 
                            env.decrease_links_utilization_sp_init(source, middlepoint, source, dest)
                            del env.sp_middlepoints[str(source)+':'+str(dest)] 

                        else: 
                            env.decrease_links_utilization_sp(source, dest, source, dest)

                        evalState = self.get_value_sp(env, source, dest, action)  
                        if evalState > nextVal:
                            nextVal = evalState
                            next_state = (action, source, dest)
                        

                        if middlepoint != -1:

                            bw = env.allocate_to_destination_sp_init(source, middlepoint, source, dest)

                            env.sp_middlepoints[str(source)+':'+str(dest)] = middlepoint
                        else:

                            env.allocate_to_destination_sp(source, dest, source, dest)
        return nextVal, next_state    
                         
    def explore_neighbourhood_DRL_sp(self, env, tmid, choice):  
        dem_iter = 0
        nextVal = -1000000
        next_state = None

        for elem in env.list_eligible_demands:
            source = elem[0]
            dest = elem[1]
            for action in range(len(env.src_dst_k_middlepoints[source][dest])):
                
                middlepoint = -1
                
                if str(source)+':'+str(dest) in env.sp_middlepoints:
                    middlepoint = env.sp_middlepoints[str(source)+':'+str(dest)] 
                    env.decrease_links_utilization_sp_init(source, middlepoint, source, dest)

                    del env.sp_middlepoints[str(source)+':'+str(dest)] 

                else: 

                    env.decrease_links_utilization_sp(source, dest, source, dest)

                evalState = self.get_value_sp(env, source, dest, action)
                if evalState > nextVal:
                    nextVal = evalState
                    next_state = (action, source, dest)
                

                if middlepoint!=-1:

                    env.allocate_to_destination_sp_init(source, middlepoint, source, dest)

                    env.sp_middlepoints[str(source)+':'+str(dest)] = middlepoint
                else:

                    env.allocate_to_destination_sp(source, dest, source, dest)

        return nextVal, next_state


def play_sp_hill_climbing_games(tm_id, srnode, choice): 

    env_hill_climb = gym.make(ENV_SIMM_ANEAL_AGENT)
    env_hill_climb.seed(SEED)
    env_hill_climb.generate_environment(general_dataset_folder, graph_topology_name, EPISODE_LENGTH_MIDDROUT, NUM_ACTIONS, percentage_demands,rate)
    currentVal = env_hill_climb.reset_hill_sp(tm_id, srnode, choice) 
    hill_climb_agent = HILL_CLIMBING(env_hill_climb)
    start = tt.time()
    while 1: 
        nextVal, next_state = hill_climb_agent.explore_neighbourhood_sp(env_hill_climb)  

        if nextVal<=currentVal or (abs((-1)*nextVal-(-1)*currentVal)<1e-4):  
            break
        

        action = next_state[0]
        source = next_state[1]
        dest = next_state[2]
       

        if str(source)+':'+str(dest) in env_hill_climb.sp_middlepoints:
            middlepoint = env_hill_climb.sp_middlepoints[str(source)+':'+str(dest)]
            env_hill_climb.decrease_links_utilization_sp_init(source, middlepoint, source, dest)

            del env_hill_climb.sp_middlepoints[str(source)+':'+str(dest)] 

        else:

            env_hill_climb.decrease_links_utilization_sp(source, dest, source, dest)
        

        currentVal = env_hill_climb.step_hill_sp(action, source, dest)
    end = tt.time()
    return currentVal*(-1), end-start

def play_DRL_GNN_sp_hill_climbing_games(tm_id, best_routing, list_of_demands_to_change, srnode, choice,linkratio):  

    env_hill_climb = gym.make(ENV_SIMM_ANEAL_AGENT)
    env_hill_climb.seed(SEED)
    env_hill_climb.generate_environment(general_dataset_folder, graph_topology_name, EPISODE_LENGTH_MIDDROUT, NUM_ACTIONS, percentage_demands,rate)
    env_hill_climb.num_critical_links = int(env_hill_climb.numEdges *linkratio)


    currentVal = env_hill_climb.reset_DRL_hill_sp(tm_id, best_routing, list_of_demands_to_change, srnode, choice)

    hill_climb_agent = HILL_CLIMBING(env_hill_climb)
    start = tt.time()
    num = 0
    while 1:
        nextVal, next_state = hill_climb_agent.explore_neighbourhood_DRL_sp(env_hill_climb, tm_id, choice)



        if nextVal<=currentVal or (abs((-1)*nextVal-(-1)*currentVal)<1e-4):
            
            break
        

        action = next_state[0]
        source = next_state[1]
        dest = next_state[2]
       

        if str(source)+':'+str(dest) in env_hill_climb.sp_middlepoints:
            middlepoint = env_hill_climb.sp_middlepoints[str(source)+':'+str(dest)]
            env_hill_climb.decrease_links_utilization_sp_init(source, middlepoint, source, dest)

            del env_hill_climb.sp_middlepoints[str(source)+':'+str(dest)] 

        else:

            env_hill_climb.decrease_links_utilization_sp(source, dest, source, dest)
        

        currentVal = env_hill_climb.step_hill_sp(action, source, dest)

    end = tt.time()


    return currentVal*(-1), end-start

class SAPAgent: 
    def __init__(self, env):
        self.K = env.K

    def act(self, env, demand, n1, n2):  
        pathList = env.allPaths[str(n1) +':'+ str(n2)]
        path = 0
        allocated = 0 
        while allocated==0 and path < len(pathList) and path<self.K:
            currentPath = pathList[path]
            can_allocate = 1 
            i = 0
            j = 1


            while j < len(currentPath):
                link_capacity = env.links_bw[currentPath[i]][currentPath[j]]
                if (env.edge_state[env.edgesDict[str(currentPath[i]) + ':' + str(currentPath[j])]][0] + demand)/link_capacity > 1:
                    can_allocate = 0
                    break
                i = i + 1
                j = j + 1

            if can_allocate==1:
                return path
            path = path + 1

        return -1

def play_sap_games(tm_id):
    env_sap = gym.make(ENV_SAP_AGENT)
    env_sap.seed(SEED)
    env_sap.generate_environment(general_dataset_folder, graph_topology_name, EPISODE_LENGTH_MIDDROUT, NUM_ACTIONS)

    demand, source, destination = env_sap.reset(tm_id)
    sap_Agent = SAPAgent(env_sap)

    rewardAddTest = 0
    start = tt.time()
    while 1:
        action = sap_Agent.act(env_sap, demand, source, destination)  #尝试

        done, error_eval_links, demand, source, destination, maxLinkUti, minLinkUti, utiStd = env_sap.step(action, demand, source, destination)
        if done:
            break
    end = tt.time()
    return maxLinkUti[2], end-start


def read_max_load_link(standard_out_file):
    pre_optim_max_load_link, post_optim_max_load_link = 0, 0
    with open(standard_out_file) as fd:
        while (True):
            line = fd.readline()
            if line.startswith("pre-optimization"):
                camps = line.split(" ")
                pre_optim_max_load_link = float(camps[-1].split('\n')[0])
            elif line.startswith("post-optimization"):
                camps = line.split(" ")
                post_optim_max_load_link = float(camps[-1].split('\n')[0])
                break
        return (pre_optim_max_load_link, post_optim_max_load_link)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parse file and create plots')

    parser.add_argument('-t', help='DEFO demands TM file id', type=str, required=True, nargs='+')  
    parser.add_argument('-g', help='graph topology name', type=str, required=True, nargs='+')  
    parser.add_argument('-m', help='model id whose weights to load', type=str, required=True, nargs='+')  
    parser.add_argument('-o', help='Where to store the pckl file', type=str, required=True, nargs='+') 
    parser.add_argument('-d', help='differentiation string', type=str, required=True, nargs='+') 
    parser.add_argument('-f', help='general dataset folder name', type=str, required=True, nargs='+')
    parser.add_argument('-f2', help='specific dataset folder name', type=str, required=True, nargs='+')  
    parser.add_argument('-r', help='sr ratio', type=str, required=True, nargs='+')
    parser.add_argument('-r1', help='link ratio', type=str, required=True, nargs='+')
    parser.add_argument('-r2', help='flow ratio', type=str, required=True, nargs='+')
    args = parser.parse_args()

    drl_eval_res_folder = args.o[0]
    tm_id = int(args.t[0])
    model_id = args.m[0]
    differentiation_str = args.d[0]
    graph_topology_name = args.g[0]
    general_dataset_folder = args.f[0]
    specific_dataset_folder = args.f2[0]
    rate = float(args.r[0])
    linkratio = float(args.r1[0])
    flowratio = float(args.r2[0])
    percentage_demands = flowratio
    results = np.zeros(100, dtype=object)

    env_help = gym.make(ENV_MIDDROUT_AGENT_SP)
    env_help.seed(SEED)

    env_help.generate_environment(general_dataset_folder, graph_topology_name, EPISODE_LENGTH_MIDDROUT, NUM_ACTIONS, percentage_demands, 1)
    env_help.top_K_critical_demands = True
    edgenum = env_help.numEdges
    env_help.num_critical_links = int(edgenum *linkratio)


    help_Agent = PPOMIDDROUTING_SP(env_help)
    checkpoint_dir = "./models" + differentiation_str
    checkpoint3 = tf.train.Checkpoint(model=help_Agent.actor, optimizer=help_Agent.optimizer)

    checkpoint3.restore(checkpoint_dir + "/ckpt_ACT-" + str(model_id))
    choice = 3
    
    sr_node, ospf_node = play_help_games_sp(repretm, env_help, help_Agent, choice, rate)


    env_drllp = gym.make(ENV_MIDDROUT_AGENT_SP)
    env_drllp.seed(SEED)
    env_drllp.generate_environment(general_dataset_folder, graph_topology_name, EPISODE_LENGTH_MIDDROUT, NUM_ACTIONS, percentage_demands, 1)
    env_drllp.top_K_critical_demands = True
    env_drllp.num_critical_links = int(edgenum *linkratio)


    DRL_LP_Agent = PPOMIDDROUTING_SP(env_drllp)
    checkpoint_dir = "./models" + differentiation_str
    checkpoint = tf.train.Checkpoint(model=DRL_LP_Agent.actor, optimizer=DRL_LP_Agent.optimizer)

    checkpoint.restore(checkpoint_dir + "/ckpt_ACT-" + str(model_id))



    env_drllp2 = gym.make(ENV_MIDDROUT_AGENT_SP)
    env_drllp2.seed(SEED)
    env_drllp2.generate_environment(general_dataset_folder, graph_topology_name, EPISODE_LENGTH_MIDDROUT, NUM_ACTIONS, percentage_demands, 1)
    env_drllp2.top_K_critical_demands = True
    env_drllp2.num_critical_links = int(edgenum *linkratio)

    DRL_LP_Agent2 = PPOMIDDROUTING_SP(env_drllp2)
    checkpoint_dir2 = "./models" + differentiation_str
    checkpoint2 = tf.train.Checkpoint(model=DRL_LP_Agent2.actor, optimizer=DRL_LP_Agent2.optimizer)

    checkpoint2.restore(checkpoint_dir2 + "/ckpt_ACT-" + str(model_id))

    env_drllp3 = gym.make(ENV_MIDDROUT_AGENT_SP)
    env_drllp3.seed(SEED)
    env_drllp3.generate_environment(general_dataset_folder, graph_topology_name, EPISODE_LENGTH_MIDDROUT, NUM_ACTIONS, percentage_demands, 1)
    env_drllp3.top_K_critical_demands = True
    env_drllp3.num_critical_links = int(edgenum *linkratio)

    DRL_LP_Agent3 = PPOMIDDROUTING_SP(env_drllp3)
    checkpoint_dir3 = "./models" + differentiation_str
    checkpoint3 = tf.train.Checkpoint(model=DRL_LP_Agent3.actor, optimizer=DRL_LP_Agent3.optimizer)

    checkpoint3.restore(checkpoint_dir3 + "/ckpt_ACT-" + str(model_id))


    env_help2 = gym.make(ENV_MIDDROUT_AGENT_SP)
    env_help2.seed(SEED)
    env_help2.generate_environment(general_dataset_folder, graph_topology_name, EPISODE_LENGTH_MIDDROUT, NUM_ACTIONS, percentage_demands, 1)
    env_help2.top_K_critical_demands = True
    env_help2.num_critical_links = int(edgenum *linkratio)

    help_Agent2 = PPOMIDDROUTING_SP(env_help2)
    checkpoint_dir4 = "./models" + differentiation_str
    checkpoint4 = tf.train.Checkpoint(model=help_Agent2.actor, optimizer=help_Agent2.optimizer)

    checkpoint4.restore(checkpoint_dir4 + "/ckpt_ACT-" + str(model_id))
    choice = 3
    drlmll_node = play_help_drl_mll_sp(repretm, env_help2, help_Agent, choice, rate)


    env_drlmll = gym.make(ENV_MIDDROUT_AGENT_SP)
    env_drlmll.seed(SEED)
    env_drlmll.generate_environment(general_dataset_folder, graph_topology_name, EPISODE_LENGTH_MIDDROUT, NUM_ACTIONS, percentage_demands, 1)
    env_drlmll.top_K_critical_demands = True
    env_drlmll.num_critical_links = int(edgenum *linkratio)

    DRL_LP_Agent5 = PPOMIDDROUTING_SP(env_drlmll)
    checkpoint_dir5 = "./models" + differentiation_str
    checkpoint5 = tf.train.Checkpoint(model=DRL_LP_Agent5.actor, optimizer=DRL_LP_Agent5.optimizer)

    checkpoint5.restore(checkpoint_dir5 + "/ckpt_ACT-" + str(model_id))

    timesteps=list()

    choice = 1
# ospf

    max_DRL, optim_cost_DRL_GNN, OSPF_init, best_routing1, list_of_demands_to_change, time_start_DRL, srnode= play_middRout_games_sp(tm_id, env_drllp, DRL_LP_Agent, 3, sr_node)

    max_link_uti_DRL_SP_HILL, optim_cost_DRL_HILL= play_DRL_GNN_sp_hill_climbing_games(tm_id, best_routing1, list_of_demands_to_change, ospf_node, choice, linkratio) 


# mine
    choice = 1

    max_DRL2, optim_cost_DRL_GNN2, OSPF_init2, best_routing2, list_of_demands_to_change2, time_start_DRL2, srnode2 = play_middRout_games_sp(tm_id, env_drllp2, DRL_LP_Agent2, 3, sr_node)

    max_link_uti_DRL_SP_HILL2, optim_cost_DRL_HILL2 = play_DRL_GNN_sp_hill_climbing_games(tm_id, best_routing2, list_of_demands_to_change2, sr_node, choice,linkratio)  


# drlmll
    choice = 1
    
    max_DRL2, optim_cost_DRL_GNN2, OSPF_init2, best_routing2, list_of_demands_to_change2, time_start_DRL2, srnode2 = play_middRout_games_sp(tm_id, env_drlmll, DRL_LP_Agent5, 3, drlmll_node)

    max_link_uti_DRL_SP_HILL2, optim_cost_DRL_HILL2 = play_DRL_GNN_sp_hill_climbing_games(tm_id, best_routing2, list_of_demands_to_change2, drlmll_node, choice,linkratio) 

# drl
    max_DRL3, optim_cost_DRL_GNN3, OSPF_init3, best_routing3, list_of_demands_to_change3, time_start_DRL3, srnode3 = play_middRout_games_sp(tm_id, env_drllp3, DRL_LP_Agent3, 1, sr_node)
# ls
    max_link_uti_sp_hill_climb2, optim_cost_HILL2 = play_sp_hill_climbing_games(tm_id, sr_node, 1)