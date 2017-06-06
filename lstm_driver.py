
import lstm
import recurrent_network
import learning_methods

file_name ="/home/lie/lol.txt"

lang = recurrent_network.compute_language(file_name)


hl_size = 100
num_hl  = 1
max_time_step = 25

net = lstm.LSTM(\
                hl_size, \
                num_hl,  \
                max_time_step, \
                lang \
)
epochs = 100
eta=2
learning_method = learning_methods.AdaGrad(eta)

net.no_batch_learn( \
                file_name, \
                epochs, \
                learning_method, \
)
