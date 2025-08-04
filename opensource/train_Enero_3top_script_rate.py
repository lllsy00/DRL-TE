import numpy as np
import gym
import gc
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import gym_graph
import random
import criticPPO as critic
import actorPPOmiddR as actor
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from collections import deque
#import time as tt
import argparse
import pickle
import heapq
from keras import backend as K
import warnings
warnings.filterwarnings("ignore")



os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

ENV_NAME = 'GraphEnv-v16'

EPISODE_LENGTH = 100 
SEED = 9
MINI_BATCH_SIZE = 64 
experiment_letter = "_B_NEW"
take_critic_demands = True 
percentage_demands = 10
str_perctg_demands = str(percentage_demands)
percentage_demands /= 100

topo_name="Biznet_"

EVALUATION_EPISODES = 10 
PPO_EPOCHS = 6
num_samples_top1 = int(np.ceil(percentage_demands*812))*4


BUFF_SIZE = num_samples_top1 

DECAY_STEPS = 60 
DECAY_RATE = 0.96

CRITIC_DISCOUNT = 0.8

ENTROPY_BETA = 0.01
ENTROPY_STEP = 60

clipping_val = 0.1
gamma = 0.99
lmbda = 0.96

max_grad_norm = 0.5

differentiation_str = topo_name+str_perctg_demands+experiment_letter
checkpoint_dir = "./models"+differentiation_str

os.environ['PYTHONHASHSEED']=str(SEED)
np.random.seed(SEED)
random.seed(SEED)

tf.random.set_seed(1)

global_step = 0
NUM_ACTIONS = 100 

hidden_init_actor = tf.keras.initializers.Orthogonal(gain=np.sqrt(2), seed=SEED)
kernel_init_actor = tf.keras.initializers.Orthogonal(gain=np.sqrt(0.01), seed=SEED)
hidden_init_critic = tf.keras.initializers.Orthogonal(gain=np.sqrt(2), seed=SEED)
kernel_init_critic = tf.keras.initializers.Orthogonal(gain=np.sqrt(1), seed=SEED)

hparams = {
    'l2': 0.0001,
    'link_state_dim': 20,
    'readout_units': 20,
    'learning_rate': 0.0002,
    'T': 5,
}

def old_cummax(alist, extractor):
    with tf.name_scope('cummax'):
        maxes = [tf.reduce_max(extractor(v)) + 1 for v in alist]
        cummaxes = [tf.zeros_like(maxes[0])]
        for i in range(len(maxes) - 1):
            cummaxes.append(tf.math.add_n(maxes[0:i + 1]))
    return cummaxes

def decayed_learning_rate(step):
    lr = hparams['learning_rate']*(DECAY_RATE ** (step / DECAY_STEPS))
    if lr<10e-5:
        lr = 10e-5
    return lr

class PPOActorCritic:
    def __init__(self):
        self.memory = deque(maxlen=BUFF_SIZE)
        self.inds = np.arange(BUFF_SIZE)
        self.listQValues = None
        self.softMaxQValues = None
        self.global_step = global_step

        self.action = None
        self.softMaxQValues = None
        self.listQValues = None

        self.utilization_feature = None
        self.bw_allocated_feature = None

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=hparams['learning_rate'], beta_1=0.9, epsilon=1e-05)
        self.actor = actor.myModel(hparams, hidden_init_actor, kernel_init_actor)
        self.actor.build()

        self.critic = critic.myModel(hparams, hidden_init_critic, kernel_init_critic)
        self.critic.build()
    
    def pred_action_distrib_sp(self, env, source, destination):

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
        paddings = tf.constant([[0, 0], [0, hparams['link_state_dim'] - 3]])
        link_state = tf.pad(tensor=hiddenStates, paddings=paddings, mode="CONSTANT")

        inputs = {'link_state': link_state, 'first': sample['first'][0:sample['length']],
                'second': sample['second'][0:sample['length']], 'num_edges': sample['num_edges']}

        return inputs

    def critic_get_graph_features(self, env):
        """
        We iterate over the converted graph nodes and take the features. The capacity and bw allocated features
        are normalized on the fly.
        """
        self.utilization_feature = env.edge_state[:,0]

        sample = {
            'num_edges': env.numEdges,  
            'length': env.firstTrueSize,
            'capacity': env.link_capacity_feature,
            'utilization': tf.convert_to_tensor(value=np.divide(self.utilization_feature, env.edge_state[:,1]), dtype=tf.float32),
            'first': env.first,
            'second': env.second
        }

        sample['utilization'] = tf.reshape(sample['utilization'][0:sample['num_edges']], [sample['num_edges'], 1])
        sample['capacity'] = tf.reshape(sample['capacity'][0:sample['num_edges']], [sample['num_edges'], 1])

        hiddenStates = tf.concat([sample['utilization'], sample['capacity']], axis=1)
        paddings = tf.constant([[0, 0], [0, hparams['link_state_dim'] - 2]])
        link_state = tf.pad(tensor=hiddenStates, paddings=paddings, mode="CONSTANT")

        inputs = {'link_state_critic': link_state, 'first_critic': sample['first'][0:sample['length']],
                'second_critic': sample['second'][0:sample['length']], 'num_edges_critic': sample['num_edges']}

        return inputs
    
    def _write_tf_summary(self, actor_loss, critic_loss, final_entropy):
        with summary_writer.as_default():
            tf.summary.scalar(name="actor_loss", data=actor_loss, step=self.global_step)
            tf.summary.scalar(name="critic_loss", data=critic_loss, step=self.global_step)  
            tf.summary.scalar(name="entropy", data=-final_entropy, step=self.global_step)                      

            tf.summary.histogram(name='ACTOR/FirstLayer/kernel:0', data=self.actor.variables[0], step=self.global_step)
            tf.summary.histogram(name='ACTOR/FirstLayer/bias:0', data=self.actor.variables[1], step=self.global_step)
            tf.summary.histogram(name='ACTOR/kernel:0', data=self.actor.variables[2], step=self.global_step)
            tf.summary.histogram(name='ACTOR/recurrent_kernel:0', data=self.actor.variables[3], step=self.global_step)
            tf.summary.histogram(name='ACTOR/bias:0', data=self.actor.variables[4], step=self.global_step)
            tf.summary.histogram(name='ACTOR/Readout1/kernel:0', data=self.actor.variables[5], step=self.global_step)
            tf.summary.histogram(name='ACTOR/Readout1/bias:0', data=self.actor.variables[6], step=self.global_step)
            tf.summary.histogram(name='ACTOR/Readout2/kernel:0', data=self.actor.variables[7], step=self.global_step)
            tf.summary.histogram(name='ACTOR/Readout2/bias:0', data=self.actor.variables[8], step=self.global_step)
            tf.summary.histogram(name='ACTOR/Readout3/kernel:0', data=self.actor.variables[9], step=self.global_step)
            tf.summary.histogram(name='ACTOR/Readout3/bias:0', data=self.actor.variables[10], step=self.global_step)
            
            tf.summary.histogram(name='CRITIC/FirstLayer/kernel:0', data=self.critic.variables[0], step=self.global_step)
            tf.summary.histogram(name='CRITIC/FirstLayer/bias:0', data=self.critic.variables[1], step=self.global_step)
            tf.summary.histogram(name='CRITIC/kernel:0', data=self.critic.variables[2], step=self.global_step)
            tf.summary.histogram(name='CRITIC/recurrent_kernel:0', data=self.critic.variables[3], step=self.global_step)
            tf.summary.histogram(name='CRITIC/bias:0', data=self.critic.variables[4], step=self.global_step)
            tf.summary.histogram(name='CRITIC/Readout1/kernel:0', data=self.critic.variables[5], step=self.global_step)
            tf.summary.histogram(name='CRITIC/Readout1/bias:0', data=self.critic.variables[6], step=self.global_step)
            tf.summary.histogram(name='CRITIC/Readout2/kernel:0', data=self.critic.variables[7], step=self.global_step)
            tf.summary.histogram(name='CRITIC/Readout2/bias:0', data=self.critic.variables[8], step=self.global_step)
            tf.summary.histogram(name='CRITIC/Readout3/kernel:0', data=self.critic.variables[9], step=self.global_step)
            tf.summary.histogram(name='CRITIC/Readout3/bias:0', data=self.critic.variables[10], step=self.global_step)
            summary_writer.flush()
            self.global_step = self.global_step + 1
    
    @tf.function(experimental_relax_shapes=True)
    def _critic_step(self, ret, link_state_critic, first_critic, second_critic, num_edges_critic):
        ret = tf.stop_gradient(ret)

        value = self.critic(link_state_critic, first_critic, second_critic,
                    num_edges_critic, training=True)[0]
        critic_sample_loss = K.square(ret - value)
        return critic_sample_loss
    
    @tf.function(experimental_relax_shapes=True)
    def _actor_step(self, advantage, old_act, old_policy_probs, link_state, graph_id, \
                    first, second, num_edges):
        adv = tf.stop_gradient(advantage)
        old_act = tf.stop_gradient(old_act)
        old_policy_probs = tf.stop_gradient(old_policy_probs)

        r = self.actor(link_state, graph_id, first, second, num_edges, training=True)
        qvalues = tf.reshape(r, (1, len(r)))
        newpolicy_probs = tf.nn.softmax(qvalues)
        newpolicy_probs2 = tf.math.reduce_sum(old_act * newpolicy_probs[0])

        ratio = K.exp(K.log(newpolicy_probs2) - K.log(tf.math.reduce_sum(old_act*old_policy_probs)))
        surr1 = -ratio*adv
        surr2 = -K.clip(ratio, min_value=1 - clipping_val, max_value=1 + clipping_val) * adv
        loss_sample = tf.maximum(surr1, surr2)

        entropy_sample = -tf.math.reduce_sum(K.log(newpolicy_probs) * newpolicy_probs[0])
        return loss_sample, entropy_sample

    def _train_step_combined(self, inds):
        entropies = []
        actor_losses = []
        critic_losses = []

        with tf.GradientTape() as tape:
            for minibatch_ind in inds:
                sample = self.memory[minibatch_ind]

                loss_sample, entropy_sample = self._actor_step(sample["advantage"], sample["old_act"], sample["old_policy_probs"], \
                            sample["link_state"], sample["graph_id"], sample["first"], sample["second"], sample["num_edges"])
                actor_losses.append(loss_sample)
                entropies.append(entropy_sample)

                critic_sample_loss = self._critic_step(sample["return"], sample["link_state_critic"], sample["first_critic"], sample["second_critic"], sample["num_edges_critic"])
                critic_losses.append(critic_sample_loss)
        
            critic_loss = tf.math.reduce_mean(critic_losses)
            final_entropy = tf.math.reduce_mean(entropies)
            actor_loss = tf.math.reduce_mean(actor_losses) - ENTROPY_BETA * final_entropy
            total_loss = actor_loss + critic_loss
            
        if str(actor_loss.numpy()) == 'nan':
            print('actor loss = nan!')
        else:
            grad = tape.gradient(total_loss, sources=self.actor.trainable_weights + self.critic.trainable_weights)

            grad, _grad_norm = tf.clip_by_global_norm(grad, max_grad_norm)
            self.optimizer.apply_gradients(zip(grad, self.actor.trainable_weights + self.critic.trainable_weights))
            entropies.clear()
            actor_losses.clear()
            critic_losses.clear()
        return actor_loss, critic_loss, final_entropy

    def ppo_update(self, actions, actions_probs, tensors, critic_features, returns, advantages):

        for pos in range(0, int(BUFF_SIZE)):

            tensor = tensors[pos]
            critic_feature = critic_features[pos]
            action = actions[pos]
            ret_value = returns[pos]
            adv_value = advantages[pos]
            action_dist = actions_probs[pos]
            
            final_tensors = ({
                'graph_id': tensor['graph_id'],
                'link_state': tensor['link_state'],
                'first': tensor['first'],
                'second': tensor['second'],
                'num_edges': tensor['num_edges'],
                'link_state_critic': critic_feature['link_state_critic'],
                'old_act': tf.convert_to_tensor(action, dtype=tf.float32),
                'advantage': tf.convert_to_tensor(adv_value, dtype=tf.float32),
                'old_policy_probs': tf.convert_to_tensor(action_dist, dtype=tf.float32),
                'first_critic': critic_feature['first_critic'],
                'second_critic': critic_feature['second_critic'],
                'num_edges_critic': critic_feature['num_edges_critic'],
                'return': tf.convert_to_tensor(ret_value, dtype=tf.float32),
            })      

            self.memory.append(final_tensors)  
        fl = True

        for i in range(PPO_EPOCHS):
            if fl == True:
                np.random.shuffle(self.inds)

                for start in range(0, BUFF_SIZE, MINI_BATCH_SIZE):
                    end = start + MINI_BATCH_SIZE
                    actor_loss, critic_loss, final_entropy = self._train_step_combined(self.inds[start:end])
                    if str(actor_loss.numpy()) == 'nan':
                        fl = False
                        break
        
        self.memory.clear()

        gc.collect()
        return actor_loss, critic_loss

def get_advantages(values, masks, rewards):
    returns = []
    gae = 0
    for i in reversed(range(len(rewards))):
        delta = rewards[i] + gamma * values[i + 1] * masks[i] - values[i]
        gae = delta + gamma * lmbda * masks[i] * gae
        returns.insert(0, gae + values[i])

    adv = np.array(returns) - values[:-1]

    return returns, (adv - np.mean(adv)) / (np.std(adv) + 1e-10)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Parse file and create plots')

    parser.add_argument('-i', help='iters', type=int, required=True)
    parser.add_argument('-c', help='counter model', type=int, required=True)
    parser.add_argument('-e', help='episode iterations', type=int, required=True)
    parser.add_argument('-f1', help='dataset folder name topology 1', type=str, required=True, nargs='+')
    args = parser.parse_args()

    dataset_folder_name1 = "../srh_datasets/"+args.f1[0]

    env_training1 = gym.make(ENV_NAME)
    env_training1.seed(SEED)
    env_training1.generate_environment(dataset_folder_name1+"/TRAIN", "Biznet", EPISODE_LENGTH, NUM_ACTIONS, percentage_demands)
    env_training1.top_K_critical_demands = take_critic_demands


    env_eval = gym.make(ENV_NAME)
    env_eval.seed(SEED)
    env_eval.generate_environment(dataset_folder_name1+"/EVALUATE", "Biznet", EPISODE_LENGTH, NUM_ACTIONS, percentage_demands)
    env_eval.top_K_critical_demands = take_critic_demands


    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    fileLogs = open("./Logs/exp" + differentiation_str + "Logs.txt", "a")

    if os.path.exists("./tmp/" + differentiation_str + "tmp.pckl"):
        f = open("./tmp/" + differentiation_str + "tmp.pckl", 'rb')
        max_reward, hparams['learning_rate'] = pickle.load(f)
        f.close()
    else:
        max_reward = -1000

    if args.i%DECAY_STEPS==0:
        hparams['learning_rate'] = decayed_learning_rate(args.i)

    if args.i>=ENTROPY_STEP:
        ENTROPY_BETA = ENTROPY_BETA/10

    agent = PPOActorCritic()

    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint_actor = tf.train.Checkpoint(model=agent.actor, optimizer=agent.optimizer)
    checkpoint_critic = tf.train.Checkpoint(model=agent.critic, optimizer=agent.optimizer)

    if args.i>0:
        checkpoint_actor = tf.train.Checkpoint(model=agent.actor, optimizer=agent.optimizer)
        checkpoint_actor.restore(checkpoint_dir + "/ckpt_ACT-" + str(args.c-1))
        checkpoint_critic = tf.train.Checkpoint(model=agent.critic, optimizer=agent.optimizer)
        checkpoint_critic.restore(checkpoint_dir + "/ckpt_CRT-" + str(args.c-1))

    reward_id = 0
    evalMeanReward = 0
    counter_store_model = args.c

    rewards_test = np.zeros(EVALUATION_EPISODES)
    error_links = np.zeros(EVALUATION_EPISODES)
    max_link_uti = np.zeros(EVALUATION_EPISODES)
    min_link_uti = np.zeros(EVALUATION_EPISODES)
    uti_std = np.zeros(EVALUATION_EPISODES)

    training_tm_ids = set(range(100))

    for iters in range(args.e):
        flag = False

        print("DRL-TE ROUTING"+experiment_letter+") PPO EPISODE: ", args.i+iters)
        while flag == False:
            states = []
            critic_features = []
            tensors = []
            actions = []
            values = []
            masks = []
            rewards = []
            actions_probs = []
            number_samples_reached = False
            tm_id = random.sample(training_tm_ids, 1)[0]
            while not number_samples_reached:

                demand, source, destination = env_training1.reset(tm_id)
                while 1:

                    tf.random.set_seed(6)

                    action_dist, tensor = agent.pred_action_distrib_sp(env_training1, source, destination)

                    features = agent.critic_get_graph_features(env_training1)

                    q_value = agent.critic(features['link_state_critic'], features['first_critic'], features['second_critic'],
                            features['num_edges_critic'], training=False)[0].numpy()[0]

                    action = np.random.choice(len(action_dist), p=action_dist)
                    action_onehot = tf.one_hot(action, depth=len(action_dist), dtype=tf.float32).numpy()

                    reward, done, _, new_demand, new_source, new_destination, _, _, _ = env_training1.step(action, demand, source, destination)
                    mask = not done

                    states.append((env_training1.edge_state, demand, source, destination))
                    tensors.append(tensor)
                    critic_features.append(features)
                    actions.append(action_onehot)
                    values.append(q_value)
                    masks.append(mask)
                    rewards.append(reward)
                    actions_probs.append(action_dist)

                    demand = new_demand
                    source = new_source
                    destination = new_destination

                    if len(states) == num_samples_top1:
                        number_samples_reached = True
                        break

                    if done:
                        break

            features = agent.critic_get_graph_features(env_training1)
            q_value = agent.critic(features['link_state_critic'], features['first_critic'], features['second_critic'],
                    features['num_edges_critic'], training=False)[0].numpy()[0]       
            values.append(q_value)

            returns, advantages = get_advantages(values, masks, rewards)
            actor_loss, critic_loss = agent.ppo_update(actions, actions_probs, tensors, critic_features, returns, advantages)
            if str(actor_loss.numpy()) == 'nan':
                print('loss nan!')
                flag = False
                continue
            else:
                flag = True

            fileLogs.write("a," + str(actor_loss.numpy()) + ",\n")
            fileLogs.write("c," + str(critic_loss.numpy()) + ",\n")
            fileLogs.flush()
            print('eval')

            for eps in range(EVALUATION_EPISODES):
                tm_id = eps
                demand, source, destination = env_eval.reset(tm_id)
                done = False
                rewardAddTest = 0
                while 1:
                    action_dist, _ = agent.pred_action_distrib_sp(env_eval, source, destination)
                    
                    action = np.argmax(action_dist)
                    reward, done, error_eval_links, demand, source, destination, maxLinkUti, minLinkUti, utiStd = env_eval.step(action, demand, source, destination)
                    rewardAddTest += reward
                    if done:
                        break
                rewards_test[eps] = rewardAddTest
                error_links[eps] = error_eval_links
                max_link_uti[eps] = maxLinkUti[2]
                min_link_uti[eps] = minLinkUti
                uti_std[eps] = utiStd
        

        evalMeanReward = np.mean(rewards_test)
        fileLogs.write(";," + str(np.mean(uti_std)) + ",\n")
        fileLogs.write("+," + str(np.mean(error_links)) + ",\n")
        fileLogs.write("<," + str(np.amax(max_link_uti)) + ",\n")
        fileLogs.write(">," + str(np.amax(min_link_uti)) + ",\n")
        fileLogs.write("ENTR," + str(ENTROPY_BETA) + ",\n")
        fileLogs.write("REW," + str(evalMeanReward) + ",\n")
        fileLogs.write("train ID," + str(counter_store_model) + ",\n")
        fileLogs.write("lr," + str(hparams['learning_rate']) + ",\n")
  
        if evalMeanReward>max_reward:
            max_reward = evalMeanReward
            reward_id = counter_store_model
            fileLogs.write("MAX REWD: " + str(max_reward) + " REWD_ID: " + str(reward_id) +",\n")
        
        fileLogs.flush()

        checkpoint_actor.save(checkpoint_prefix+'_ACT')
        checkpoint_critic.save(checkpoint_prefix+'_CRT')
        counter_store_model = counter_store_model + 1
        K.clear_session()
        gc.collect()

    f = open("./tmp/" + differentiation_str + "tmp.pckl", 'wb')
    pickle.dump((max_reward, hparams['learning_rate']), f)
    f.close()

