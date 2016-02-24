
# coding: utf-8

# In[197]:

import numpy as np
from numpy import exp, tanh, log, cosh

def sigmoid(x):
    return 1.0/(1.0+exp(-x))

def d_sigmoid(x):
    return exp(x)/(exp(x)+1.0)/(exp(x)+1.0)

def reLU(x):
    return max(0.0,x)

def d_reLU(x):
    if x>0:
        return 1.0
    else:
        return 0.0

def softplus(x):
    return log(1.0+exp(x))

def d_softplus(x):
    return 1.0/(1.0+exp(-x))

def d_tanh(x):
    return 4.0*cosh(x)*cosh(x)/(cosh(2.0*x)+1)/(cosh(2.0*x)+1)
    
class neuron:
    def __init__(self, n_weights, activation_type='sigmoid', bias=0):   #supports linear, reLU, softplus, or sigmoid activations
        self.n_weights=np.array(n_weights)
        self.activation_type=activation_type
        self.bias=bias
        self.weights=np.random.rand(n_weights)
        
    def change_weights(self, weights_vector):
        weights_vector=list(weights_vector)
        if len(weights_vector)==self.n_weights:
            self.weights=np.array(weights_vector)
        else:
            print 'Failed to change neuron connection strengths, weights vector has incorrect format'
            
    def print_weights(self):
        print self.weights
        
    def activate(self, input_vector):
        input_vector=np.array(input_vector)
        if input_vector.size==self.n_weights:
            input_vector=np.array(input_vector)
            if self.activation_type=='sigmoid':
                self.output=sigmoid(sum(self.weights*input_vector))+self.bias
                return self.output
            elif self.activation_type=='tanh':
                self.output=tanh(sum(self.weights*input_vector))+self.bias
                return self.output
            elif self.activation_type=='reLU':
                self.output=reLU(sum(self.weights*input_vector))+self.bias
                return self.output
            elif self.activation_type=='linear':
                self.output=float(sum(self.weights*input_vector))+self.bias
                return self.output
            elif self.activation_type=='softplus':
                self.output=softplus(sum(self.weights*input_vector))+self.bias
                return self.output
            else:
                print 'Unknown activation function: '+self.activation_type
        else:
            print 'Activation input format incorrect'
            
    def backprop(self, input_vector, step_size=0.01):
        if self.activation_type=='sigmoid':
            self.gradient=self.weights*d_sigmoid(sum(self.weights*input_vector))
        elif self.activation_type=='tanh':
            self.gradient=self.weights*d_tanh(sum(self.weights*input_vector))
        elif self.activation_type=='reLU':
            self.gradient=self.weights*d_reLU(sum(self.weights*input_vector))
        elif self.activation_type=='linear':
            self.gradient=self.weights*sum(self.weights*input_vector)
        elif self.activation_type=='softplus':
            self.gradient=self.weights*d_softplus(sum(self.weights*input_vector))
        else:
            print 'Unknown activation function: '+self.activation_type
            return 0
        self.weights=self.weights+step_size*self.gradient
        return self.weights

class layer(neuron):
    def __init__(self, n_neurons, n_weights_per_neuron, activation_type='sigmoid', bias_vector=0):
        self.n_neurons=n_neurons
        self.activation_type=activation_type
        self.n_weights_per_neuron=n_weights_per_neuron
        if bias_vector==0:
            bias_vector=np.zeros(n_neurons)  
        if len(bias_vector)!=n_neurons:
            print 'Layer initialization failed, check length of bias vector'        
        i=0
        self.layer=[]
        while(i<n_neurons):
            self.layer.append(neuron(n_weights_per_neuron, activation_type, bias_vector[i]))
            i+=1            
        #print 'Layer created'
    
    def change_weights(self, weights_vector):
        weights_vector=list(weights_vector)
        if len(weights_vector)==self.n_neurons and len(weights_vector[0])==self.n_weights_per_neuron:
            for i in range(len(self.layer)):                
                self.layer[i].change_weights(weights_vector[i])
        else:
            print 'Failed to change layer connection strengths, weights vector has incorrect format'
        
    def print_weights(self):
        for i in self.layer:
            print i.weights
            
    def activate(self, input_vector):
        input_vector=np.array(input_vector)
        output=[]
        if input_vector.ndim==2:
            if input_vector.shape[0]!=self.n_neurons or input_vector.shape[1]!=self.n_weights_per_neuron:
                print 'Incorrect input vector in layer activation function'
                return output
        elif input_vector.ndim!=self.n_weights_per_neuron:
            print 'Incorrect input vector in layer activation function'
            return output
        for n in range(len(self.layer)):
            output.append(self.layer[n].activate(input_vector[n]))
        return output
    
class net():
    n_layers=0
    def __init__(self, connection_type='full'):
        self.connection_type=connection_type
        self.net=[]
        #print 'Network created'
    
    def addlayer(self, layer):
        if len(self.net)==0:
            if layer.n_weights_per_neuron!=1:
                print 'Addlayer failed, input layer should be added first with 1 weight per neuron'
                return 0
            else:
                self.net.append(layer)
                self.n_layers+=1
        else:
            i=len(self.net)
            if self.connection_type=='full' and layer.n_weights_per_neuron==self.net[i-1].n_neurons:                   
                self.net.append(layer)
                self.n_layers+=1
            else:
                print 'Number of inputs per neuron is incorrect in the added layer, layer not added'         
        
    def activate(self, input_vector):
        if len(input_vector)!=self.net[0].n_neurons:
            print 'Activation failed. Input vector of incorrect length'
            return 0
        if self.net[0].n_weights_per_neuron!=1:
            print 'Activation failed. Input layer should have 1 weight per neuron!'
            return 0
        out=self.net[0].activate(input_vector)
        for i in range(1,len(self.net)):
            inp=[]
            for n in range(self.net[i].n_neurons):
                inp.append(out)
            out=self.net[i].activate(inp)
        return out


# In[202]:

l1=layer(10, 1, activation_type='linear')

l2=layer(200, 10, activation_type='sigmoid')

l3=layer(100, 200, activation_type='sigmoid')

l4=layer(1, 100, activation_type='linear')

n=net()
n.addlayer(l1)
n.addlayer(l2)
n.addlayer(l3)
n.addlayer(l4)

#print n.activate([1,1,1,1,1,1,1,1,1,1])

n1=neuron(2,activation_type='sigmoid')
print n1.activate([1,-2])
n1.print_weights()
n1.backprop([1,-2], step_size=0.1)
print n1.gradient
print n1.activate([1,-2])

