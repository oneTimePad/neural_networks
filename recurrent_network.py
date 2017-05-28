"""
Multi-layer Vanilla Recurrent Neural Network (simple implementation)
based on https://gist.github.com/karpathy/d4dee566867f8291f086

"""

import numpy as np
import activation_functions
import cost_functions

def compute_language(lang_file):
    """
        create dictionaries from language file
        from repo reference above
    """
    f = open(lang_file,'r')
    data =f.read()
    f.close()
    #collect unique characters
    chars = list(set(data))
    data_size,vocab_size = len(data),len(chars)
    print('data of size %d has %d unique characters' %(data_size,vocab_size))
    return ({ch:i for i,ch in enumerate(chars)},{i:ch for i,ch in enumerate(chars)},vocab_size)



class RNNetwork(object):
    """
    implements multi-layer vanilla RNN
    """
    def __init__(self,hl_size,num_hl,max_time_step,language):
        """
        hl_size:= size of the hidden layer state vector
        num_hl:= number of hidden layers (vertically)
        max_time_step:= maximum input size for RNN
        language:= dictionaries for the network language
        """

        #the dictionaries for the language
        self.char_to_ix,self.ix_to_char,self.vocab_size = language

        #weight matrix for the input layer
        first_W = np.random.randn(hl_size,self.vocab_size+hl_size)
        self.hl_size = hl_size
        self.num_hl = num_hl
        self.max_time_step = max_time_step
        #consists of all weight matrices for all num_hl hiddens layers
        self.weights_hidden=[first_W*.01]+[np.random.randn(hl_size,2*hl_size)*.01 for i in range(0,num_hl-1)]
        #bias for all hidden layers
        self.biases_hidden = [np.zeros((hl_size,1)) for i in range(0,num_hl)]
        #only one output layer
        self.weights_output = np.random.randn(self.vocab_size,hl_size)*.01
        self.biases_output  = np.zeros((self.vocab_size,1))
        #saved states
        #[{layer1,layer2....},{layer_1,layer2,...},...]
        self.hidden_states =[[np.zeros((hl_size,1)) for m in range(0,num_hl)] for j in range(0,max_time_step)]
        #the iniital hidden states for all layers
        self.initial_state = [np.zeros((hl_size,1)) for m in range(0,num_hl)]
        #saved outputs
        self.outputs =  {}
        #saved outputs before softmax applied
        self.z = {}



    def reset_memory(self):
        #[{layer1,layer2....},{layer_1,layer2,...},...]
        self.hidden_states =[[np.zeros((self.hl_size,1)) for m in range(0,self.num_hl)] for j in range(0,self.max_time_step)]
        #the iniital hidden states for all layers
        self.initial_state = [np.zeros((self.hl_size,1)) for m in range(0,self.num_hl)]
        #saved outputs
        self.outputs =  {}
        #saved outputs before softmax applied
        self.z = {}


    def forward_pass(self,inputs,initial_state=None):
        """
        forward pass through the network
        inputs:= list of numeric representations of elements of the language
        returns list of numeric representations of elements of the language predicted at each output
        returns the output and also updates internal states of network
        """

        initial_state = initial_state if initial_state!=None else self.initial_state

        #go through each time step
        for i in range(0,min(len(inputs),self.max_time_step)):
            #go through each layer
            #compute the input at the ith time step
            one_hot = np.zeros((self.vocab_size,1))
            one_hot[inputs[i]] =1
            hidden_states_i = self.hidden_states[i-1] if i!=0 else initial_state
            for j,actions in enumerate(zip(self.weights_hidden,self.biases_hidden,hidden_states_i)):
                #extract the hidden_weights,biases, and previous state for this layer
                #w: weight matrix for the jth layer
                #b: bias for the jth layer
                #prev: previous hidden state for the jth layer in the ith time-step
                w,b,prev_hidden = actions
                #state_ij = tanh(hj*(state_i-1j,state_i,j-1) +bj)
                input_state = self.hidden_states[i][j-1] if j!=0 else one_hot
                self.hidden_states[i][j] = np.tanh(np.dot(w,np.concatenate((prev_hidden,input_state)))+b)
            #after going through all layers, use the last hidden state to compute the output
            #could be made way more efficient,just making it explicit
            self.z[i] = np.dot(self.weights_output,self.hidden_states[i][-1])+self.biases_output
            self.outputs[i] =activation_functions.Softmax().transform(self.z[i])
        return [self.ix_to_char[np.argmax(o)] for o in self.outputs]

    def backward_pass(self,inputs,targets):
        """
        performs backprop in time for multi-layer RNN
        performs backprop downward and then left
        """
        #go backwards in time
        #gradient for the output weights
        dWy = np.zeros_like(self.weights_output)
        dby = np.zeros_like(self.biases_output)
        #this accounts for both Whh and Whx at each layer
        #each layer has a [Whh Whx] tensor
        dWh = [ np.zeros_like(w) for w in self.weights_hidden]
        dbh = [ np.zeros_like(b) for b in self.biases_hidden ]
        #used for backward pass (left) columns are vertical backward passes
        dhback = [np.zeros_like(w) for w in self.biases_hidden]
        for i in reversed(range(0,len(inputs))):
            #dWy can be calculate very efficiently , just done this way to make it explicit
            #target vector for input i
            y = np.zeros_like(self.outputs[i])
            y[targets[i]]=1
            #compute the input at the ith time step
            one_hot = np.zeros((self.vocab_size,1))
            one_hot[inputs[i]] =1
            #dE_ij/dy_ij := half_delta (the error derivative)
            dEy =cost_functions.CrossEntropy().derivative(self.outputs[i],y)* \
                            activation_functions.Softmax().derivative(self.z[i])
            #dE_ij/dh_ij:= outer product with last layer hidden states
            dWy+=np.outer(dEy,self.hidden_states[i][-1])
            dby+=dEy
            #dE_ij/dh_ij = Wy_T (dEy) back into h_ij
            half_delta =np.dot(self.weights_output.transpose(),dEy)
            #last layer hidden state dE_i/dz_ij
            # dE_ij/dz_ij = dE_ij/d_hji *dh_ij/dz_ij (unit error, same as feedforward NN)
            delta = half_delta*(1.0-self.hidden_states[i][-1]**2)
            #dh = dE_i/dz_ij  + dE_i+1/dz_ij + ... dE_len(input)-1/dz_ij
            dh  = delta + dhback[-1]*(1.0-self.hidden_states[i][-1]**2)
            #fetch states for this timestep at each layer
            hsi = self.hidden_states[i]
            #highest layer hidden bias
            dbh[-1]+=dh
            #Wh = [Wh Wx] thus dWh = [delta (outer) hi-1,j delta(outer)h[i][j-1] ]
            input_state = self.hidden_states[i][-2] if len(self.weights_hidden)>1 else one_hot
            prev_hidden_state = self.hidden_states[i-1][-1]
            dWh[-1]+=np.concatenate((np.dot(dh,prev_hidden_state.transpose()),\
                    np.dot(delta,input_state.transpose())),axis=1)
            #backpass to the left the cumuative sum  of dE_i's (make them all with respec to the next hidden layer to the left)
            #Whh_T*dh
            dhback[-1] = np.dot(self.weights_hidden[-1][:,:self.hl_size].transpose(),dh)
            #need to repeat this for all lower layers
            for j in range(2,len(self.weights_hidden)+1):
                #never need to back into input so its ok
                #fetch Whx(x means the input to the state, this is for multi-layer RNN)
                #backpass downwards
                #back into prev layer state (dE_i/dz_ij (i:=timestep,j:=layer))
                #Whx_T*delta (pass delta down layers)
                delta = np.dot(self.weights_hidden[-j+1][:,self.hl_size:].transpose(),delta)*(1.0-hsi[-j]**2)
                #cumulative dE_i's with respect to the zij layer
                dh = delta + dhback[-j]*(1.0-hsi[-j]**2)
                dbh[-j]+=dh
                prev_hidden_state = self.hidden_states[i-1][-j]
                input_state = hsi[-j-1]  if j!=len(self.weights_hidden) else one_hot
                dWh[-j]+=np.concatenate((np.dot(dh,prev_hidden_state.transpose()),\
                        np.dot(delta,input_state.transpose())),axis=1)

                whh = self.weights_hidden[-j][:,:self.hl_size]
                dhback[-j] =np.dot(whh,dh)
        #clip to stop exploding gradient
        np.clip(dWy,-5,5,out=dWy)
        np.clip(dby,-5,5,out=dby)
        dWh = [ np.clip(w,-5,5) for w in dWh]
        dbh = [ np.clip(b,-5,5) for b in dbh]

        return dWy,dWh,dby,dbh

    def compute_mini_batch_gradients(self,mini_batch):
        """
        compute mini-batch gradients like in feedforward NN
        Not used right now
        """
        dWy,dWh,dby,dbh = np.zeros_like(self.weights_output),[np.zeros_like(w) for w in self.weights_output], \
                            np.zeros_like(self.biases_output), [np.zeros_like(b) for b in self.biases]
        for x,y in mini_batch:
            outputs = self.forward_pass(x)
            grad_Wy,grad_Wh,grad_by,grad_bh = self.backward_pass(x,y)
            dWy = [cw + dw for cw,dw in zip(dWy,grad_Wy)]
            dWh = [cw + dw for cw,dw in zip(dWh,grad_Wh)]
            dby = [cb + db for cb,db in zip(dby,grad_by)]
            dbh = [cb + db for cb,db in zip(dbh,grad_bh)]
        dWy = [cw/len(mini_batch) for cw in dWy]
        dWh = [cw/len(mini_batch) for cw in dWh]
        dby = [cb/len(mini_batch) for cb in dby]
        dbh = [cb/len(mini_batch) for cb in dbh]
        return dWy,dWh,dby,dbh

    def sample_sequence(self,seed_ix,n):
        """
        samples a sequence of the NN:  code from reference above
        """
        #one hot representation
        ixs = []
        h = self.initial_state
        x = seed_ix
        for t in range(0,n):
            #evaluates first time step of forward pass with init state h
            self.forward_pass([x],h)
            #evaluate a single time step in the forward pass( we don't need the others now)
            #just keep setting the initial_state from the prev state
            x = np.random.choice(range(0,self.vocab_size),p=self.outputs[0].ravel())
            #fetch the first step hidden state to feed into the next
            h = self.hidden_states[0]
            ixs.append(x)
        return ixs



    def no_batch_learn(self,training_data,epochs,learning_method):
        """
        perform non-mini-batch learning
        just simply goes through the input sequence by sequence
        from reference above
        """
        f = open(training_data,'r')
        data = f.read()
        f.close()
        for j in range(epochs):
            p = 0
            n = 0
            while p+self.max_time_step+1 < len(data):
                inputs = [self.char_to_ix[ch] for ch in data[p:p+self.max_time_step]]
                targets = [self.char_to_ix[ch] for ch in data[p+1:p+self.max_time_step+1]]
                if n% 100 == 0:
                    samples = self.sample_sequence(inputs[0],200)
                    txt = ''.join(self.ix_to_char[ix] for ix in samples)
                    print('----\n %s \n----' % (txt,))

                outputs = self.forward_pass(inputs)
                self.initial_state = self.hidden_states[self.max_time_step-1]
                dWy,dWh,dby,dbh = self.backward_pass(inputs,targets)
                learning_method.update((self.weights_output,dWy),(self.biases_output,dby))
                learning_method.update((self.weights_hidden,dWh),(self.biases_hidden,dbh))

                p+=self.max_time_step
                n+=1
            self.reset_memory()
