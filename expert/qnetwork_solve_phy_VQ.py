# THIS VERSION IS SLIGHT MODIFIED IN ORDER TO SOLVE V AND Q FOR PHYSICIAN POLICIES.
# INSTEAD OF TAKING MAX, WE TAKE MEAN, NAMELY, double_q_value = np.mean(Q2, axis=1)
import tensorflow as tf
import numpy as np
import math
import os
import random
import numpy as np
import pandas as pd
from pandas import DataFrame
import math
import copy
import pickle
from sklearn.model_selection import train_test_split
import time
from optparse import OptionParser

parser = OptionParser()
parser.add_option("--reward_type", dest="reward_type", default=1)
parser.add_option("--num_steps", dest="num_steps", default=100000)
parser.add_option("--reg_lambda", dest="reg_lambda", default=5)

opts,args = parser.parse_args()

reward_type = int(opts.reward_type)
reg_lambda = float(opts.reg_lambda)


REWARD_THRESHOLD = 3
reg_lambda = 5


hidden_1_size = 128
hidden_2_size = 128

class Qnetwork():
	def __init__(self):
		
		self.phase = tf.placeholder(tf.bool)
		self.num_actions = 25
		self.input_size = 128
		self.state = tf.placeholder(tf.float32, shape=[None, self.input_size],name="input_state")
		
		self.fc_1 = tf.contrib.layers.fully_connected(self.state, hidden_1_size, activation_fn=None)
		self.fc_1_bn = tf.contrib.layers.batch_norm(self.fc_1, center=True, scale=True, is_training=self.phase)
		self.fc_1_ac = tf.maximum(self.fc_1_bn, self.fc_1_bn * 0.5)
		
		self.fc_2 = tf.contrib.layers.fully_connected(self.fc_1_ac, hidden_2_size, activation_fn=None)
		self.fc_2_bn = tf.contrib.layers.batch_norm(self.fc_2, center=True, scale=True, is_training=self.phase)
		self.fc_2_ac = tf.maximum(self.fc_2_bn, self.fc_2_bn * 0.5)
		
		# advantage and value streams
		self.streamA, self.streamV = tf.split(self.fc_2_ac,2,axis=1)
		self.Advantage = tf.contrib.layers.fully_connected(self.streamA, self.num_actions, activation_fn=None)
		self.Value = tf.contrib.layers.fully_connected(self.streamV, 1, activation_fn=None)
		
		# Then combine them together to get our final Q-values.
		self.q_output = self.Value + tf.subtract(self.Advantage, tf.reduce_mean(self.Advantage,axis=1,keep_dims=True))
		self.predict = tf.argmax(self.q_output, 1, name='predict') # vector of length batch size
		
		# Below we obtain the loss by taking the sum of squares difference between the target and predicted Q values.
		self.targetQ = tf.placeholder(shape=[None],dtype=tf.float32)
		self.actions = tf.placeholder(shape=[None],dtype=tf.int32)
		self.actions_onehot = tf.one_hot(self.actions, self.num_actions,dtype=tf.float32)
		
		# Importance sampling weights for PER, used in network update    
		self.imp_weights = tf.placeholder(shape=[None], dtype=tf.float32)
		# select the Q values for the actions that would be selected
		self.Q = tf.reduce_sum(tf.multiply(self.q_output, self.actions_onehot), reduction_indices=1) # batch size x 1 vector
		
		# reward threshold, to ensure reasonable Q-value predictions  
		self.reg_vector = tf.maximum(tf.abs(self.Q)-REWARD_THRESHOLD,0)
		self.reg_term = tf.reduce_sum(self.reg_vector)
		
		self.abs_error = tf.abs(self.targetQ - self.Q)
		
		self.td_error = tf.square(self.targetQ - self.Q)
		
		# below is the loss when we are not using PER
		self.old_loss = tf.reduce_mean(self.td_error)
		
		# as in the paper, to get PER loss we weight the squared error by the importance weights
		self.per_error = tf.multiply(self.td_error, self.imp_weights)

		# total loss is a sum of PER loss and the regularisation term
		if per_flag:
			self.loss = tf.reduce_mean(self.per_error) + reg_lambda * self.reg_term
		else:
			self.loss = self.old_loss + reg_lambda * self.reg_term

		self.trainer = tf.train.AdamOptimizer(learning_rate=0.0001)
		self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
		with tf.control_dependencies(self.update_ops):
		# Ensures that we execute the update_ops before performing the model update, so batchnorm works
			self.update_model = self.trainer.minimize(self.loss)

# function is needed to update parameters between main and target network
# tf_vars are the trainable variables to update, and tau is the rate at which to update
# returns tf ops corresponding to the updates
def update_target_graph(tf_vars,tau):
	total_vars = len(tf_vars)
	op_holder = []
	for idx,var in enumerate(tf_vars[0:int(total_vars/2)]):
		op_holder.append(tf_vars[idx+int(total_vars/2)].assign((var.value()*tau) + ((1-tau)*tf_vars[idx+int(total_vars/2)].value())))
	return op_holder

def update_target(op_holder,sess):
	for op in op_holder:
		sess.run(op)

# define an action mapping - how to get an id representing the action from the (iv,vaso) tuple
action_map = {}
count = 0
for iv in range(5):
	for vaso in range(5):
		action_map[(iv, vaso)] = count
		count += 1

# generates batches for the Q network - depending on train and eval_type, can select data from train/val/test sets.
def process_batch(size, train_phase=True, eval_type = None):
	
	if not train_phase:
		
		if eval_type is None:
			raise Exception('Provide eval_type to process_batch')
		elif eval_type == 'train':
			a = all_train.copy()
		elif eval_type == 'val':
			a = val.copy()
		elif eval_type == 'test':
			a = test.copy()
		else:
			raise Exception('Unknown eval_type')
	else:
		if per_flag:
			# uses prioritised exp replay
			a = train.sample(n=size, weights=train['prob'])
		else:
			a = train.sample(n=size)
			
	if size == None:
		size = len(a)
	
	states = np.zeros((size, 128))
	actions = np.zeros((size, 1), dtype=int)
	rewards = np.zeros((size, 1))
	next_states = np.zeros((size, 128))
	done_flags = np.zeros((size, 1))

	if reward_type == 0 and eval_type != 'test':
		reward = obser.loc['reward']
	
	counter = 0
	for idx, obser in a.iterrows():
		cur_state = obser[:128]
		iv = int(obser.loc['iv_input'])
		vaso = int(obser.loc['vaso_input'])
		action = action_map[iv, vaso]
		
		if idx != all_train.index[-1]:
			# if not terminal step in trajectory             
			if all_train.loc[idx, 'icustayid'] == all_train.loc[idx + 1, 'icustayid']:
				if reward_type == 1 and eval_type != 'test':
					reward = all_train.loc[idx + 1, 'reward'] - obser.loc['reward']
				next_state = all_train.iloc[idx + 1, :128]
				done = 0
			else:
				# trajectory is finished
				next_state = np.zeros(len(cur_state))
				if reward_type == 1:
					reward = 0
				done = 1
		else:
			# last entry in df is the final state of that trajectory
			next_state = np.zeros(len(cur_state))
			done = 1
		
		states[counter] = cur_state
		actions[counter] = action
		if eval_type != 'test':
			rewards[counter] = reward
		next_states[counter] = next_state
		done_flags[counter] = done
		counter += 1
		
	return (states, np.squeeze(actions), np.squeeze(rewards), next_states, np.squeeze(done_flags), a)

def evaluate(s, a, r, s_, done_flags):

	# firstly get the chosen actions at the next timestep
	actions_from_q1 = sess.run(mainQN.predict, feed_dict={mainQN.state: s_, mainQN.phase : 0})

	# Q values for the next timestep from target network, as part of the Double DQN update
	Q2 = sess.run(targetQN.q_output,feed_dict={targetQN.state:s_, targetQN.phase : 0})
	# handles the case when a trajectory is finished
	end_multiplier = 1 - done_flags

	# target Q value using Q values from target, and actions from main
	# double_q_value = Q2[range(len(actions_from_q1)), actions_from_q1]
	double_q_value = np.mean(Q2, axis=1)

	# definition of target Q
	targetQ = r + ( gamma * double_q_value * end_multiplier )

	# get the output q's, actions, and loss
	q_output, actions_taken, abs_err = sess.run([mainQN.q_output, mainQN.predict, mainQN.abs_error], \
		feed_dict={mainQN.state:s,
				   mainQN.targetQ:targetQ, 
				   mainQN.actions:a,
				   mainQN.phase:False})
	# return the relevant q values and actions
	# phys_q = q_output[range(len(q_output)), a]
	# agent_q = q_output[range(len(q_output)), actions_taken]
	
	return q_output, actions_taken, a


def do_eval(eval_type, batch=False, batch_size=128):

	states,actions,rewards,next_states,done_flags, _ = process_batch(size=None,train_phase=False,eval_type=eval_type)
	
	if batch:

		num_batches = states.shape[0] // batch_size
		
		Q = np.zeros((states.shape[0], 25))
		agent_actions = np.zeros(actions.shape)
		
		for batch in range(num_batches+1):

			batch_state = states[batch*batch_size: (batch+1)*batch_size]
			batch_actions = actions[batch*batch_size: (batch+1)*batch_size]
			batch_rewards = rewards[batch*batch_size: (batch+1)*batch_size]
			batch_next_s = next_states[batch*batch_size: (batch+1)*batch_size]
			batch_done_flags = done_flags[batch*batch_size: (batch+1)*batch_size]

			q_output, agent_a, phy_actions = evaluate(batch_state, batch_actions, batch_rewards, batch_next_s, batch_done_flags)

			Q[batch*batch_size: (batch+1)*batch_size] = q_output
			agent_actions[batch*batch_size: (batch+1)*batch_size] = agent_a

	else:

		Q, agent_actions, phy_actions = evaluate(states,actions,rewards,next_states,done_flags)
	
	return Q, agent_actions, phy_actions



if __name__ == '__main__':

	print ('reading data ...')
	all_train = pickle.load(open('data/dqn_train_set.pkl', 'rb')) #pd.read_csv('data/dqn_train_set.csv')
	test =  pickle.load(open('data/dqn_test_set.pkl', 'rb')) #pd.read_csv('data/dqn_test_set.csv')
	print ('reading data completed')

	# PER important weights and params
	per_flag = True
	beta_start = 0.9
	all_train['prob'] = abs(all_train['reward'])
	temp = 1.0 / all_train['prob']
	temp[temp == float('Inf')] = 1.0
	all_train['imp_weight'] = pow((1.0 / len(all_train) * temp), beta_start)

	uids = np.unique(all_train['icustayid'])
	train_uids, val_uids = train_test_split(uids, test_size=0.2, random_state=42)
	train = all_train[all_train['icustayid'].isin(train_uids)]
	val = all_train[all_train['icustayid'].isin(val_uids)]


	tf.set_random_seed(3)
	np.random.seed(3)
	# The main training loop is here
	per_alpha = 0.6 # PER hyperparameter
	per_epsilon = 0.01 # PER hyperparameter
	batch_size = 30 #How many experiences to use for each training step.
	gamma = 0.99 #Discount factor on the target Q-values
	num_steps = int(opts.num_steps)
	load_model = False #Whether to load a saved model.
	save_dir = '../../data/dqn/'
	save_path = "./model/"#The path to save our model to.
	tau = 0.001 #Rate to update target network toward primary network
	tf.reset_default_graph()
	mainQN = Qnetwork()
	targetQN = Qnetwork()
	av_q_list = []
	save_results = False

	saver = tf.train.Saver(tf.global_variables())

	init = tf.global_variables_initializer()
	trainables = tf.trainable_variables()
	target_ops = update_target_graph(trainables, tau)

	#Make a path for our model to be saved in.
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)

	loss_hist = []
	mean_q_hist = [] # (step, phy_q, agent_q)

	with tf.Session() as sess:
		if load_model == True:
			print('Trying to load model...')
			try:
				restorer = tf.train.import_meta_graph(save_path + 'ckpt.meta')
				restorer.restore(sess, tf.train.latest_checkpoint(save_dir))
				print ("Model restored")
			except IOError:
				print ("No previous model found, running default init")
				sess.run(init)
			try:
				per_weights = pickle.load(open( save_dir + "per_weights.p", "rb" ))
				imp_weights = pickle.load(open( save_dir + "imp_weights.p", "rb" ))
				
				# the PER weights, governing probability of sampling, and importance sampling
				# weights for use in the gradient descent updates
				train['prob'] = per_weights
				train['imp_weight'] = imp_weights
				print ("PER and Importance weights restored")
			except IOError:
				print("No PER weights found - default being used for PER and importance sampling")
		else:
			print("Running default init")
			sess.run(init)
		print("Init done")
		
		for i in range(num_steps):
			
			if save_results:
				do_save_results()
				break
				
			net_loss = 0.0
			net_q = 0.0
			
			states, actions, rewards, next_states, done_flags, sampled_df = process_batch(batch_size)
			# firstly get the chosen actions at the next timestep
			actions_from_q1 = sess.run(mainQN.predict, feed_dict={mainQN.state:next_states, mainQN.phase : 1})
			# actions chosen now, as a check
			cur_act = sess.run(mainQN.predict, feed_dict={mainQN.state:states, mainQN.phase : 1})
			
			# Q values for the next timestep from target network, as part of the Double DQN update
			Q2 = sess.run(targetQN.q_output, feed_dict={targetQN.state:next_states, targetQN.phase : 1})
			# handles the case when a trajectory is finished
			end_multiplier = 1 - done_flags
			
			# target Q value using Q values from target, and actions from main
			# double_q_value = Q2[range(batch_size), actions_from_q1]
			double_q_value = np.mean(Q2, axis=1)
			
			# empirical hack to make the Q values never exceed the threshold - helps learning
			double_q_value[double_q_value > REWARD_THRESHOLD] = REWARD_THRESHOLD
			double_q_value[double_q_value < -REWARD_THRESHOLD] = -REWARD_THRESHOLD
			
			# definition of target Q
			targetQ = rewards + (gamma * double_q_value * end_multiplier)
			
			# Calculate the importance sampling weights for PER
			imp_sampling_weights = np.array(sampled_df['imp_weight'] / float(max(train['imp_weight'])))
			imp_sampling_weights[np.isnan(imp_sampling_weights)] = 1
			imp_sampling_weights[imp_sampling_weights <= 0.001] = 0.001
			
			# Train with the batch
			_, loss, error, q_output = sess.run([mainQN.update_model, mainQN.loss, mainQN.abs_error, mainQN.q_output], \
				feed_dict={mainQN.state: states,
						   mainQN.targetQ: targetQ, 
						   mainQN.actions: actions,
						   mainQN.phase: True,
						   mainQN.imp_weights: imp_sampling_weights})
			
			
			update_target(target_ops, sess)
			
			net_loss += sum(error)
			net_q += np.mean(targetQ)
			
			# Set the selection weight/prob to the abs prediction error and update the importance sampling weight
			new_weights = pow((error + per_epsilon), per_alpha)
			train.loc[train.index.isin(sampled_df.index), 'prob'] = new_weights
			temp = 1.0/new_weights
			train.loc[train.index.isin(sampled_df.index), 'imp_weight'] = pow(((1.0/len(train)) * temp), beta_start)
			
			if i % 500 == 0 and i > 0:
				saver.save(sess, save_path)
				print("Saved Model, step is " + str(i))
				
				av_loss = net_loss / 500.0
				loss_hist += [ av_loss ]
				print("Average loss is ", av_loss)
				net_loss = 0.0
				 
				print ("Saving PER and importance weights")
				with open(save_dir + 'per_weights.p', 'wb') as f:
					pickle.dump(train['prob'], f)
				with open(save_dir + 'imp_weights.p', 'wb') as f:
					pickle.dump(train['imp_weight'], f)
			
				print ('step:', i)
				print ("phys actions: ", actions)
				print ("chosen actions: ", cur_act)
				if i % 500 == 0:
					# run an evaluation on the validation set
					q_output, agent_actions, phys_actions = do_eval(eval_type = 'val')
					phys_q = q_output[range(len(q_output)), phys_actions]
					agent_q = q_output[range(len(q_output)), agent_actions]
	#               print ('mean abs err:', mean_abs_error)
					print ('mean phys Q:', np.mean(phys_q))
					print ('mean agent Q:', np.mean(agent_q))
					mean_q_hist += [(i, np.mean(phys_q), np.mean(agent_q))]
				print ('------------------------')

		pickle.dump(loss_hist, open('data/outcome_phy/train_loss_dist', 'wb'))
		pickle.dump(mean_q_hist, open('data/outcome_phy/mean_Q_dist', 'wb'))
		# pickle.dump((phys_q, phys_actions, agent_q, agent_actions), open('data/outcome/val', 'wb'))
		print ('predicting train set ...')
		Q_train, agent_actions_train, _ = do_eval('train', batch=True)
		print ('predicting test set ...')
		Q_test, agent_actions_test, _ = do_eval('test', batch=True)
		pickle.dump((Q_train, agent_actions_train), open('data/outcome_phy/results_train.pkl', 'wb'))
		pickle.dump((Q_test, agent_actions_test), open('data/outcome_phy/results_test.pkl', 'wb'))