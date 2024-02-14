# batch size,Default=32
batch_size = 32

# Max length of sentence,Default=25
max_length = 20

# Generator embedding size
g_e = 64

# Generator LSTM hidden size
g_h = 64

# Discriminator embedding and Highway network sizes
d_e = 64

# Discriminator LSTM hidden size
d_h = 64

# Number of Monte Calro Search
n_sample = 16

# Number of generated sentences,Default=500,20000
generate_samples = 500

# Pretraining parameters,g_pre_epochs_Default=20,d=3
g_pre_epochs = 5
d_pre_epochs = 2

g_lr = 1e-5

# Discriminator dropout ratio
d_dropout = 0.0
d_lr = 1e-6

# Pretraining parameters
g_pre_lr = 1e-2
d_pre_lr = 1e-4
