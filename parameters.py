dims = (100, 100)
timesteps_per_batch = 1000
max_pathlength = 1000000
max_kl = 0.01
cg_damping = 0.1
gamma = 0.95
render = False
train_flag = True
# dims of observation, also is the input dims of action network
obs_shape = [None, 7]
checkpoint_file = "checkpoint/iter804.ckpt"
environment_name = "MountainCar-v0"