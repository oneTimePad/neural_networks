import tensorflow as tf
from collections import deque

height=88
width=80
channel=3

X_state = tf.placeholder(tf.float32,shape=[None,height,width,channel])


def q_network(X,scope):
	with tf.variable_scope(scope):
		conv1 = tf.layers.conv2d(X,filters=32,kernel_size=8,strides=4,padding="SAME",name="conv1_8x8")
		conv2 = tf.layers.conv2d(conv1,filters=64,kernel_size=4,strides=2,padding="SAME",name="conv2_4x4")
		conv3 = tf.layers.conv2d(conv2,filters=64,kernel_size=3,strides=1,padding="SAME",name="conv3_3x3")
		conv4 = tf.layers.conv2d(conv3,filters=512,kernel_size=4,padding="SAME",name="fc4")
		conv5 = tf.layers.conv2d(conv4,filters=9,kernel_size=1,padding="SAME",name="fc5")
		logits = tf.squeeze(conv5)
	return logits,{var.name:var for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope=scope)}

actor,actor_vars = q_network(X_state,"actor")
critic,critic_vars = q_network(X_state,"critic")

#we must copy parameters of critic to actor
copy_op = [actor_var.assign(critic_vars[name]) for name,actor_var in actor_vars.items()]
copy_op = tf.group(*copy_op)

#action and q's taken from critic
X_action = tf.placeholder(tf.int32,shape=[None])
q_value = tf.reduce_sum(critic_q_values* tf.one_hot(X_action,n_outputs))

#critic's cost: match Q-values of actor
y = tf.placeholder(tf.float32,shape=[None,1])

with tf.name_scope("critic_loss"):
	cost = tf.reduce_mean(tf.square(y-q_value))
global_step = tf.contrib.framework.get_or_create_global_step()
with tf.name_scope("critic_train"):
	optimizer =tf.train.AdamOptimizer(lr)
	critic_train_op = optimizer.minimize(cost,global_step=global_step)

init = tf.global_variables_initializer()
saver = tf.train.Saver()


replay_memory_size =10000
replay_memory = deque([],maxlen=replay_memory_size)

def sample_memories(batch_size):
	indices = rnd.permutation(len(replay_memory)[:batch_size])
	col = [[],[],[],[],[]] #state, action, reward, next_state, continue
	for idx in indices:
		memory = replay_memory[idx]
		for col, value in zip(cols,memory):
			col.append(value)
	cols = [np.array(col) for col in cols]
	return (cols[0],cols[1],cols[2].reshape(-1,1),cols[3],cols[4].reshape(-1,1))

eps_min = 0.05
eps_max = 1.0
eps_decay_steps = 50000

def epsilon_greedy(q_values,step):
	epsilon = max(eps_min, eps_max-(eps_max-eps_min)*step/eps_decay_steps)
	if rnd.rand() < epsilon:
		return rnd.randint(n_outputs)
	else:
		return np.argmax(q_values)

n_steps = 10000
training_start = 1000
training_interval = 3
save_steps = 50
copy_steps = 25
discount_rate = 0.95
skip_start = 90
batch_size = 50
iteration = 0
checkpoint_path = "./my_dqn.ckpt"
done = True


with tf.Session() as sess:
	if os.path.isfile(checkpoint_path):
		saver.restore(sess,checkpoint_path)
	else:
		init.run()
	while True:
		step = global_step.eval()
		if step >= n_steps:
			break
		iteration +=1
		if done:
			obs = env.reset()
			for skip in range(skip_start):
				obs, reward, done, info = env.step(0)
			state = preprocess_observation(obs)


		q_values = actor_q_values.eval(feed_dict={X_state:[state]})
		action = epsilon_greedy(q_values,step)

		replay_memory.append((state,action,reward,next_state,1.0-done))
		state=next_state

		if iteration < training_state or iteration % training_interval !=0:
			continue

		X_state_val, X_action_val, rewards, X_next_state_val, continues =(sample_memories(batch_size))
		next_q_values = actor_q_values.eval(feed_dict={X_state:X_next_state_val})
		max_mext_q_values = np.max(next_q_values,axis=1,keepdims=True)
		y_val = rewards + continues*discount_rate*max_next_q_values
		critic_train_op.run(feed_dict={X_state:X_state_val,X_action:X_action_val,y:y_val})

		if step %copy_steps == 0:
			copy_op.run()
		if step % save_steps == 0:
			saver.save(Sess,checkpoint_path)








