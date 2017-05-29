

import recurrent_network
import learning_methods


file_name =

lang = recurrent_network.compute_language(file_name)


hl_size = 100
num_hl  = 1
max_time_step = 25

net = recurrent_network.RNNetwork(\
                hl_size, \
                num_hl,  \
                max_time_step, \
                lang \
)

epochs = 100
eta=.01
learning_method = learning_methods.GradientDescent(eta)

net.no_batch_learn( \
                file_name, \
                epochs, \
                learning_method, \
)
