# for image
dims = (100, 100)
obs_height = 100
obs_width = 100
obs_channel = 1
history_number = 2

# for trainning
jobs = 3
max_iter_number = 10000
paths_number = 8
max_path_length = 30
max_kl = 0.01
gae_lambda = 1.0
subsample_factor = 0.8
cg_damping = 0.1
discount = 0.99
cg_iters = 10
deviation = 0.1
render = True
train_flag = False
iter_num_per_train = 1
checkpoint_file = "checkpoint/iter0.ckpt"
record_movie = False

# for environment

environment_name = "Pendulum-v0"

# for continous action
min_std = 1e-6
center_adv = True
positive_adv = False
use_std_network = False
std = 1.1
obs_shape = 3
action_shape = 1
min_a = -2.0
max_a = 2.0

