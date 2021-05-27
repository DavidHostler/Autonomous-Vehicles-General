from Buffer import Buffer
from H5 import H5




std_dev = 0.2
ou_noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(std_dev) * np.ones(1))

actor_model = get_actor()
critic_model = get_critic()

target_actor = get_actor()
target_critic = get_critic()

# Making the weights equal initially
target_actor.set_weights(actor_model.get_weights())
target_critic.set_weights(critic_model.get_weights())

# Learning rate for actor-critic models
critic_lr = 0.002
actor_lr = 0.001

critic_optimizer = tf.keras.optimizers.Adam(critic_lr)
actor_optimizer = tf.keras.optimizers.Adam(actor_lr)

total_episodes = 100
# Discount factor for future rewards
gamma = 0.99
# Used to update target networks
tau = 0.005

buffer = Buffer(50000, 64)

#buffer = Buffer(50000, 4)





def get_actor():
    # Initialize weights between -3e-3 and 3-e3
    last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

    inputs = layers.Input(shape=(num_states,))
    out = layers.Dense(256, activation="relu")(inputs)
    out = layers.Dense(256, activation="relu")(out)
    
    #Original
    #Accel_y = layers.Dense(1, activation="tanh", kernel_initializer=last_init)(out)    
    #Accel_x = layers.Dense(1, activation="tanh", kernel_initializer=last_init)(out) 
    
    
    Accel_y = layers.Dense(1, activation="sigmoid", kernel_initializer=last_init)(out)    
    Accel_x = layers.Dense(1, activation="sigmoid", kernel_initializer=last_init)(out)
    
    #outputs = layers.Dense(1, activation="tanh", kernel_initializer=last_init)(out)
    
    outputs = tf.keras.layers.Concatenate()([Accel_x, Accel_y])
    
    # Our upper bound is 2.0 for Pendulum.
    #outputs = outputs * upper_bound
    #model = tf.keras.Model(inputs, outputs)
    model = Model(inputs,outputs)
    return model

    env = H5()



def get_critic():
    # State as input
    state_input = layers.Input(shape=(num_states))
    state_out = layers.Dense(16, activation="relu")(state_input)
    state_out = layers.Dense(32, activation="relu")(state_out)

    # Action as input
    action_input = layers.Input(shape=(2))
    action_out = layers.Dense(32, activation="relu")(action_input)

    # Both are passed through seperate layer before concatenating
    concat = layers.Concatenate()([state_out, action_out])

    out = layers.Dense(256, activation="relu")(concat)
    out = layers.Dense(256, activation="relu")(out)
    outputs = layers.Dense(1)(out)

    # Outputs single value for give state-action
    model = tf.keras.Model([state_input, action_input], outputs)

    return model
  
  
  
  ep_reward_list = []
# To store reward history of each episode
ep_reward_list = []
# To store average reward history of last few episodes
avg_reward_list = []

state_history = [[], []]
state_info = np.zeros(5)

# Takes about 4 min to train
counter = 0
x_path = []
y_path = []

v_x = []
v_y = []

for ep in range(total_episodes):

    env.reset()
    prev_state = np.array([0,0,0,0])
    #prev_state = env.reset()
    original_state = prev_state
    episodic_reward = 0

    

    while True:
        # Uncomment this to see the Actor in action
        # But not in a python notebook.
        # env.render()

        tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)

        action = policy(tf_prev_state, ou_noise)
        # Recieve state and reward from environment.
        state, reward, done, info = env.step(action)

        buffer.record((prev_state, action[0]  , reward, state))
        episodic_reward += reward

        buffer.learn()
        update_target(target_actor.variables, actor_model.variables, tau)
        update_target(target_critic.variables, critic_model.variables, tau)

        # End this episode when `done` is True
        
        if done:
            break

         
             

        prev_state = state
        
        
        ep_reward_list.append(episodic_reward)
        avg_reward = np.mean(ep_reward_list[-40:])
        #state_info = np.array(ep, state, reward, done, info)
        state_history[0].append(ep)
        state_history[1].append(done)
          
        x_path.append(env.x)
        y_path.append(env.y)
        
        v_x.append(env.x_dot)
        v_y.append(env.y_dot)
        
        if len(state_history[0]) % 250 == 0:
            counter += 1
            env.reset()
            episodic_reward = 0
            print("Episode * {} * Avg Reward is ==> {}".format(counter, avg_reward))    
            
        else:
            prev_state = state
        
        #if len(ep_reward_list) % 40 == 0:
        avg_reward_list.append(avg_reward)           
        #print(counter, state, reward, done, info)
    
        #print("Episode * {} * Avg Reward is ==> {}".format(counter, avg_reward))    
        #avg_reward_list.append(avg_reward)
  
  
  
  
  
  
