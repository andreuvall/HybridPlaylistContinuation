# Song2playist classifier configuration file.

# Network structure
n_layers = 3
n_hidden = 100
hid_nl = 'tanh'
out_nl = 'sigmoid'

# Training options
batch_size = 50
learning_rate = 0.5
max_epochs = 250
momentum = True

# Regularization
input_dropout = 0.1
hidden_dropout = 0.5
positive_weight = 1.
nonpositive_weight = 1.
L1_weight = 0.
L2_weight = 0.

# Features
feature = 'songtags'
standardize = True
normalize = 'l2'
