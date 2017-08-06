import tensorflow as tf
import gym
import numpy as np
dense = tf.layers.dense

env = gym.make('CartPole-v0')
obs = env.reset()


n_inputs = 4
n_hidden = 4
n_outputs = 1
initializer = tf.contrib.layers.variance_scaling_initializer()
lr = 0.01

X = tf.placeholder(tf.float32,shape=[None,n_inputs])

hidden = dense(X,n_hidden,activation=tf.nn.elu,kernel_initializer=initializer)
logits = dense(hidden,n_outputs,activation=None,kernel_initializer=initializer)

outputs = tf.nn.sigmoid(logits)

p_left_and_right = tf.concat(axis=1,values=[outputs,1-outputs])
action = tf.multinomial(tf.log(p_left_and_right),num_samples=1)

#ouputs the target probability of left
y = 1. - tf.to_float(action)
xentropy = tf.nn.sigmoid_cross_entropy_with_logits(
            labels = y, logits=logits)
optimizer = tf.train.AdamOptimizer(lr)
grads_and_vars = optimizer.compute_gradients(xentropy)
gradients = [grad for grad,variable in grads_and_vars]
gradients_placeholders = []
grads_and_vars_feed = []
for grad,variable in grads_and_vars:
    gradients_placeholder = tf.placeholder(tf.float32,shape=grad.get_shape())
    gradients_placeholders.append(gradients_placeholder)
    grads_and_vars_feed.append((gradients_placeholder,variable))
train_op = optimizer.apply_gradients(grads_and_vars_feed)

init = tf.global_variables_initializer()

def discount_rewards(rewards, discount_rate):
    discounted_rewards = np.empty(len(rewards))
    cumulative_rewards = 0
    for step in reversed(range(len(rewards))):
        cumulative_rewards = rewards[step] + cumulative_rewards * discount_rate
        discounted_rewards[step] = cumulative_rewards

def discount_and_normalize_rewards(all_rewards,discount_rate):
    all_discounted_rewards = [discount_rewards(rewards,discount_rate) for rewards in all_rewards]
    flat_rewards = np.concatenate(all_discounted_rewards)
    reward_mean = flat_rewards.mean()
    reward_std  = flat_rewards.std()
    return [(discounted_rewards-reward_mean)/reward_std for discounted_rewards in all_discounted_rewards]


n_iterations = 250
n_max_steps = 1000
n_games_per_update = 10
discount_rate = 0.95
save_iterations = 10
saver = tf.train.Saver()
with tf.Session() as sess:
    init.run()
    for iterations in range(n_iterations):
        all_rewards = []
        all_gradients = []
        for game in range(n_games_per_update):
            current_rewards = []
            current_gradients = []
            obs = env.reset()
            for step in range(n_max_steps):
                print(step)
                action_val, gradients_val = sess.run([action,gradients],feed_dict={X:obs.reshape(1,n_inputs)})
                obs, reward, done, info = env.step(action_val[0][0])
                #env.render()
                current_rewards.append(reward)
                current_gradients.append(gradients_val)
                if done:
                    print('failed')
                    break

            all_rewards.append(current_rewards)
            all_gradients.append(current_gradients)
        feed_dict = {}
        for var_index, gradients_placeholder in enumerate(gradients_placeholders):
            mean_gradients = np.mean(
                [reward * all_gradients[game_index][step][var_index]
                        for game_index, rewards in enumerate(all_rewards)
                        for step,reward in enumerate(rewards)],
                axis=0)
            feed_dict[gradients_placeholder] = mean_gradients
        sess.run(train_op,feed_dict=feed_dict)
        if iterations % save_iterations == 0:
            saver.save(sess, "./my_policy_net_basic.ckpt")

from PIL import Image, ImageDraw
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
def update_scene(num, frames, patch):
    patch.set_data(frames[num])
    return patch,

def plot_animation(frames, repeat=False, interval=40):
    plt.close()  # or else nbagg sometimes plots in the previous cell
    fig = plt.figure()
    patch = plt.imshow(frames[0])
    plt.axis('off')
    return animation.FuncAnimation(fig, update_scene, fargs=(frames, patch), frames=len(frames), repeat=repeat, interval=interval)

try:
    from pyglet.gl import gl_info
    openai_cart_pole_rendering = True   # no problem, let's use OpenAI gym's rendering function
except Exception:
    openai_cart_pole_rendering = False  # probably no X server available, let's use our own rendering function

def render_cart_pole(env, obs):
    if openai_cart_pole_rendering:
        # use OpenAI gym's rendering function
        return env.render(mode="rgb_array")
    else:
        # rendering for the cart pole environment (in case OpenAI gym can't do it)
        img_w = 600
        img_h = 400
        cart_w = img_w // 12
        cart_h = img_h // 15
        pole_len = img_h // 3.5
        pole_w = img_w // 80 + 1
        x_width = 2
        max_ang = 0.2
        bg_col = (255, 255, 255)
        cart_col = 0x000000 # Blue Green Red
        pole_col = 0x669acc # Blue Green Red

        pos, vel, ang, ang_vel = obs
        img = Image.new('RGB', (img_w, img_h), bg_col)
        draw = ImageDraw.Draw(img)
        cart_x = pos * img_w // x_width + img_w // x_width
        cart_y = img_h * 95 // 100
        top_pole_x = cart_x + pole_len * np.sin(ang)
        top_pole_y = cart_y - cart_h // 2 - pole_len * np.cos(ang)
        draw.line((0, cart_y, img_w, cart_y), fill=0)
        draw.rectangle((cart_x - cart_w // 2, cart_y - cart_h // 2, cart_x + cart_w // 2, cart_y + cart_h // 2), fill=cart_col) # draw cart
        draw.line((cart_x, cart_y - cart_h // 2, top_pole_x, top_pole_y), fill=pole_col, width=pole_w) # draw pole
        return np.array(img)

def render_policy_net(model_path, action, X, n_max_steps = 1000):
    frames = []
    env = gym.make("CartPole-v0")
    obs = env.reset()
    with tf.Session() as sess:
        saver.restore(sess, model_path)
        for step in range(n_max_steps):
            img = render_cart_pole(env, obs)
            frames.append(img)
            action_val = action.eval(feed_dict={X: obs.reshape(1, n_inputs)})
            obs, reward, done, info = env.step(action_val[0][0])
            #env.render()
            if done:
                break
    env.close()
    return frames
frames = render_policy_net("./my_policy_net_basic.ckpt", action, X)
video = plot_animation(frames)
plt.show()
