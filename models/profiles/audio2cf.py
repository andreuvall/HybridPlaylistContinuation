# Song2playlist classifier configuration file.

# Network structure
n_layers = 3
n_hidden = 100
hid_nl = 'tanh'
out_nl = 'sigmoid'

# Training options
batch_size = 50
learning_rate = 0.5
max_epochs = 160
momentum = True

# Early-stopping options
patience = 10
refinement = 5
factor_lr = 0.5
max_epochs_increase = 1.1
significance_level = 0.95

# Regularization
input_dropout = 0.1
hidden_dropout = 0.5
positive_weight = 1.
nonpositive_weight = 1.
l1_weight = 0.
l2_weight = 0.

# Features
feature = 'audio2cf'
standardize = True
normalize = 'l2'
