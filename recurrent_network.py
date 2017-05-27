"""
Multi-layer Vanilla Recurrent Neural Network (simple implementation)
based on https://gist.github.com/karpathy/d4dee566867f8291f086

"""



class RNNetwork(object):
    """
    implements multi-layer vanilla RNN
    """
    def __init__(self,hl_size,num_hl,max_time_step,input_size,language):
        #weight matrix for the input layer
        first_W = np.random.randn(hl_size,input_size+hl_size)
        self.hl_size = hl_size
        #consists of all weight matrices for all num_hl hiddens layers
        self.weights_hidden=[first_W]+[np.random.randn(hl_size,2*hl_size) for i in range(0,num_hl-1)]
        #bias for all hidden layers
        self.biases_hidden = [self.random.randn(hl_size,1) for i in range(0,num_hl)]
        #only one output layer
        self.weights_output = self.random.randn(input_size,hl_size)
        self.biases_output  = self.random.randn(input_size,1)

        #saved states
        #[{layer1,layer2....},{layer_1,layer2,...},...]
        self.hidden_states =[ [np.random.randn(hl_size,1) for i in range(0,num_hl)] ]+[{} for j in range(1,input_size)]
        self.outputs =  {}
        self.z = {}
        self.language = language

    def forward_pass(self,inputs):
        """
        forward pass through the network
        inputs:= list of numeric representations of elements of the language
        returns list of numeric representations of elements of the language predicted at each output
        """

        #go through each time step
        for i in range(0,len(inputs)):
            #go through each layer
            #set the input to the first hidden state the i-th time_step
            one_hot = np.zeros((self.language,1))
            one_hot[inputs[i]] =1
            self.hidden_states[i][-1]=one_hot
            for j,actions in enumerate((self.weights_hidden,self.biases_hidden,self.hidden_states[i-1])):
                #extract the hidden_weights,biases, and previous state for this layer
                #h: weight matrix for the jth layer
                #b: bias for the jth layer
                #prev: previous hidden state for the jth layer in the ith time-step
                h,b,prev = actions
                #state_ij = hj*(state_i-1j,state_i,j-1) +bj
                self.hidden_states[i][j] = np.tanh(np.dot(h,np.concatenate(prev,self.hidden_states[i][j-1]))+b)
            #after going through all layers, use the last hidden state to compute the output
            #could be made way more efficient,just making it explicit
            self.z[i] = np.dot(self.weights_output,self.hidden_states[i][-1])+self.biases_output
            self.outputs[i] =activation_functions.Softmax().transform(self.z[i])
        return [self.language[np.argmax(o)] for o in self.outputs]

    def backward_pass(self,inputs,targets):
        """
        performs backprop in time for multi-layer RNN
        """
        #go backwards in time
        #gradient for the output weights
            self.weights_hidden=[first_W]+[np.random.randn(hl_size,2*hl_size) for i in range(0,num_hl-1)]
        dWy = np.zeros_like(self.weights_output)
        dby = np.zeros_like(self.biases_output)
        dWh = [ np.zeros_like(w) for w in self.weights_hidden]
        dbh = [ np.zeros_like(b) for b in self.biases_hidden ]
        #used for backward pass (left) columns are vertical backward passes
        dhback = [np.zeros_like(w) for w in self.biases_hidden]
        for i in reversed(range(0,len(inputs))):
            #dWy can be calculate very efficiently , just done this way to make it explicit
            #target vector for input i
            y = np.zeros_like(self.outputs[i])
            y[targets[i]]=1
            #dE_ij/dy_ij := half_delta
            dEy = self.CrossEntropy().derivative(self.outputs[i],y)*self.derivative(self.z[i])
            #dE_ij/dh_ij:= outer product with last layer hidden states
            dWy+=np.outer(dEy,self.hidden_states[i][-1])
            dby+=delta_output_layer
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
            dWh[-1]+=np.concatenate((np.dot(dh,self.hidden_states[i-1][-1].transpose()),\
                    np.dot(delta,self.hidden_states[i][-2]).transpose()),axis=1)
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
                delta = np.dot(self.weights_hidden[-j+1][:,self.hl_size:].transpose,delta)*(1.0-hsi[-j]*hsi[-j])
                #cumulative dE_i's with respect to the zij layer
                dh = delta + dhback[-j]*(1.0-self.hsi[-j]**2)
                dbh[-j]+=dh
                dWh[-j]+=np.concatenate((np.dot(dh,self.hidden_states[i-1][-j].transpose()),\
                        np.dot(delta,self.hsi[-j-1]).transpose()),axis=1)
                dhback[-j] =np.dot(self.weights_hidden[-j][:,self.hl_size:])
