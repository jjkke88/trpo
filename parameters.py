dims = (100, 100)
timesteps_per_batch = 1000
max_pathlength = 200
max_kl = 0.01
cg_damping = 0.1
gamma = 0.95
deviation = 0.1
PI = 3.1415926
render = True
train_flag = True
# dims of observation, also is the input dims of action network
obs_shape = 3
action_shape = 1
min_a = -2.0
max_a = 2.0
checkpoint_file = "checkpoint/iter18.ckpt"
environment_name = "Pendulum-v0"