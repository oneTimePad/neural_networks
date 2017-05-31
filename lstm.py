



class LSTM(object):

    def __init__(self,hl_size,num_hl,max_time_step,language):

        #the dictionaries for the language
        self.char_to_ix,self.ix_to_char,self.vocab_size = language

        #weight matrix for the input layer
        first_W = np.random.randn(4*hl_size,self.vocab_size+hl_size)
        #first_W = np.ones((hl_size,self.vocab_size+hl_size))
        self.hl_size = hl_size
        self.num_hl = num_hl
        self.max_time_step = max_time_step
        #consists of all weight matrices for all num_hl gate layers
        #accounts for i,f,g,o tensors
        self.weights_gates=[first_W*.01]+[np.random.randn(4*hl_size,2*hl_size)*.01 for i in range(0,num_hl-1)]
        #bias for all hidden layers
        self.biases_gates = [np.zeros((4*hl_size,1)) for i in range(0,num_hl)]
        #only one output layer
        self.weights_output = np.random.randn(self.vocab_size,hl_size)*.01
        #self.weights_output = np.ones((self.vocab_size,hl_size))*.01
        self.biases_output  = np.zeros((self.vocab_size,1))
        #saved states (cell and hidden)
        #[{layer1,layer2....},{layer_1,layer2,...},...]
        #[hidden,cell]
        self.states =[[np.zeros((hl_size,2)) for m in range(0,num_hl)] for j in range(0,max_time_step)]
        #the initial hidden/cell states for all layers
        self.initial_states = [np.zeros((hl_size,2)) for m in range(0,num_hl)]
        #cached f,g,i,o gates
        self.gates = [[np.zeros((4*hl_size,1)) for m in range(0,num_hl)] for j in range(0,max_time_step)]
        #saved outputs
        self.outputs =  {}
        #saved outputs before softmax applied
        self.temporal_affine = {}

    def reset_memory(self):
        #the initial hidden/cell states for all layers
        self.initial_states = [np.zeros((hl_size,2)) for m in range(0,num_hl)]

    #just for making things clear
    @property
    def i(self,gates):
        return gates[0:self.hl_size,:]
    @property
    def f(self,gates):
        return gates[self.hl_size:2*self.hl_size,:]
    @property
    def o(self,gates):
        return gates[2*self.hl_size:3*self.hl_size,:]
    @property
    def g(self,gates):
        return gates[3*self.hl_size:4*self.hl_size,:]
    @property
    def hidden(self,state):
        return state[:,0]
    @property
    def cell(self,state):
        return state[:,1]


    def gate_transform(self,affine_gates):
        """
        apply gate Non-Linearity
        """
        h = self.hl_size
        affine_gates[h,:] = activation_functions.Sigmoid().transform(affine_gates[h,:])
        affine_gates[2*h,:] = activation_functions.Sigmoid().transform(affine_gates[2*h,:])
        affine_gates[3*h,:] = activation_functions.Sigmoid().transform(affine_gates[3*h,:])
        affine_gates[4*h,:] = activation_functions.Tanh().transform(affine_gates[4*h,:])
        transformed_gates = affine_gates
        return transformed_gates

    def forward_pass(self,inputs,initial_state):
        """
        forward pass through the network
        inputs:= list of numeric representations of elements of the language
        initial_state (H,2)->(Hidden,Cell)
        returns list of numeric representations of elements of the language predicted at each output
        returns the output and also updates internal states of network
        """

        initial_state = initial_state if initial_state!=None else self.initial_states

        #go through each time step
        for i in range(0,min(len(inputs),self.max_time_step)):
            #go through each layer
            #compute the input at the ith time step
            one_hot = np.zeros((self.vocab_size,1))
            one_hot[inputs[i]] =1
            hidden_states_i = self.hidden_states[i-1] if i!=0 else initial_state
            for j,actions in enumerate(zip(self.weights_gates,self.biases_gates,hidden_states_i)):
                #extract the hidden_weights,biases, and previous state for this layer
                #w: weight matrix for the jth layer
                #b: bias for the jth layer
                #prev: previous state for the jth layer in the ith time-step
                w,b,prev_state = actions
                #use prev layer output
                input_state = self.states[i][j-1] if j!=0 else one_hot
                #compute combined vector of [h_i-1,j:h_i,j-1]
                combined_state = np.concatenate(self.hidden(prev_state),self.hidden(input_state))
                #compute i,f,o,g gates
                gates =  self.gate_transform(np.dot(w,combined_state)+b)
                self.gates[i][j] = gates
                #compute c_i and h_i
                new_cell = self.cell(prev_state)*self.f(gates)+self.g(gates)*self.i(gates)
                new_hidden = np.tanh(new_cell)*self.o(gates)
                #combine states to [h_i:c_i]
                self.states[i][j] = np.concatenate((new_hidden,new_cell),axis=1)
            #after going through all layers, use the last hidden state to compute the output
            #could be made way more efficient,just making it explicit
            self.temporal_affine[i] = np.dot(self.weights_output,self.hidden(self.hidden_states[i][-1]))+self.biases_output
            self.outputs[i] =activation_functions.Softmax().transform(self.z[i])

        return [self.ix_to_char[np.argmax(self.outputs[o])] for o in self.outputs.keys()]

    def compute_hidden_state_gradients(self,i,j,prev_state,input_state):
        """
        computes dh_ij/dz_ij for all gates for the i,j layer (i,f,o,g)
        i:= time-step where higher i is forward in time
        j:= layer where lower j is higher layer(reversed to make it easier in loops)
        prev_state:= previous state back in time
        input_state:= lower layer input state
        """
        #gate and state values
        gates_ij = self.gates[i][-j]
        states_ij = self.hidden_states[i][-j]



        #i-th time step gate values for all layers
        i_gate = self.i(gates_ij)
        f_gate = self.f(gates_ij)
        o_gate = self.o(gates_ij)
        g_gate = self.g(gates_ij)

        #dh_i/dc_i
        dc_ij = self.o(gates_ij)*(1-self.hidden(states_ij)**2)

        #dhi_ij/dzi_ij = dc_ij*g_ij*[i*(1-i)]
        dhi = dc_ij*g_gate*(i_gate)*(1-i_gate)
        #dh_ij/dzf_ij = dc_ij*c_i-1,j *[f*(1-f)]
        dhf = dc_ij*self.cell(prev_state)*f_gate*(1-f_gate)
        #dh_ij/dzo_ij = dh_ij/do_ij *do_i/dzo_ij = tanh(c_ij)*[o*(1-o)]
        dho = np.tanh(self.cell(states_ij))*o_gate*(1-o_gate)
        #h_ij/dzg_ij = dc_i*i_ij*(1-g**2)
        dhg = dc_ij*i_gate*(1-g_gate**2)

        #dh_ij/dz_ij
        dh_ij = np.concatenate(dhi,dhf,dho,dhg)
        return dh_ij

    def backward_pass(self,inputs,targets):
        """
        performs backprop in time for multi-layer RNN
        performs backprop downward and then left
        """
        #go backwards in time
        #gradient for the output weights
        dWy = np.zeros_like(self.weights_output)
        dby = np.zeros_like(self.biases_output)
        #this accounts for all the gate gradients
        #both for the input weight matrix and the past hidden state weight matrix
        dWg = [ np.zeros_like(w) for w in self.weights_gates]
        dbg = [ np.zeros_like(b) for b in self.biases_gates ]
        #used for backward pass (left) columns are vertical backward passes
        half_delta_back = [np.zeros_like(w) for w in self.biases_gates]


        for i in reversed(range(0,len(inputs))):
            #dWy can be calculate very efficiently , just done this way to make it explicit
            #target vector for input i
            y = np.zeros_like(self.outputs[i])
            y[targets[i]]=1
            #compute the input at the ith time step
            one_hot = np.zeros((self.vocab_size,1))
            one_hot[inputs[i]] =1
            #dE_ij/dy_ij := (the error derivative)
            dEy= np.copy(self.outputs[i])
            dEy[targets[i]]-=1

            #dE_ij/dh_ij:= outer product with last layer hidden states
            dWy+=np.dot(dEy,self.hidden(states_ij).transpose())
            dby+=dEy
            #dE_ij/dh_ij = Wy_T (dEy) back into h_ij
            half_delta =np.dot(self.weights_output.transpose(),dEy)

            #for all layers
            for j in range(1,len(self.weights_hidden)+1):
                #computer dh_ij/dz_ij for (i,f,o,g)
                dh_ij = self.compute_hidden_state_gradients(i,-j)

                #check if it is the state top layer
                if j!=1:
                    #never need to back into input so its ok
                    #fetch Whx(x means the input to the state, this is for multi-layer RNN)
                    #backpass downwards
                    #back into prev layer state (dE_i/dz_ij (i:=timestep,j:=layer))
                    #Whx_T*delta (pass delta down layers)
                    Wgx_ij_T = self.weights_gates[-j+1][:self.hl_size:].transpose()
                    delta = np.dot(Wgx_ij_T,delta)*dh_ij
                else:
                    #dE_i/dz_i = dE_i/dh_i *dh_i/dz_i
                    delta = half_delta*dh_i
                #cumulative dE_i's with respect to the zij layer
                delta_t= delta + half_delta_back*dh_ij

                dbg[-j]+=delta_t
                #prev_state,and input_state values
                prev_state = self.states[i-1][-j] if i!=0 else self.initial_state[-j]
                input_state = self.states[i][-j-1] if len(self.weights_gates)>1 else one_hot
                #for i,f,o,g hidden weight matrices
                dWgh_ij = np.dot(delta_t,self.hidden(prev_state).transpose())
                #for i,f,o,g input weight matrices
                #x represents input component of temporal_affine layer for this layer (not necessarily input, could be below hidden layer state)
                dWgx_ij = np.dot(delta_t,input_state.transpose())
                #total weight matrix gradient
                dWg_ij = np.concatenate((dWgh_ij,dWgx_ij),axis=1)
                dWg[-j]+=dWg_ij

                #backpass to the left the cumuative sum  of dE_i's (make them all with respect to the next hidden layer to the left)
                #Wgh_ij_t * delta_t (i-th step,j-th layer)
                Wgh_ij_T = self.weights_gates[-j][:,:self.hl_size].transpose()
                half_delta_back[-j] = np.dot(Wgh_ij_T,delta_t)

        #clip to stop exploding gradient
        np.clip(dWy,-5,5,out=dWy)
        np.clip(dby,-5,5,out=dby)
        dWg = [ np.clip(w,-5,5) for w in dWg]
        dbg = [ np.clip(b,-5,5) for b in dbg]

        return dWg,dWg,dby,dbh
