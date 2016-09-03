dims = (100, 100)
timesteps_per_batch = 1000
max_pathlength = 100
max_kl = 0.01
cg_damping = 0.1
gamma = 0.95
render = False
train_flag = True
# dims of observation, also is the input dims of action network
obs_height = 100
obs_width = 100
obs_channel = 1
history_number = 2
checkpoint_file = "checkpoint/iter804.ckpt"
environment_name = "CartPole-v0"