# resources:- 
#      1) https://hausetutorials.netlify.app/posts/2019-12-01-neural-networks-deriving-the-sigmoid-derivative/
#      2) https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
#      3) https://victorzhou.com/blog/intro-to-neural-networks/
import numpy as np

def activate(x):
    return 1/(1+ np.exp(-x))
    
def derivate_activation(x):
    return activate(x) * (1 - activate(x))
    
def error_loss(actual, predicted):
    return ((actual - predicted) **2).mean()
    

class NeuralNet:
    def __init__(self):
        self.w1 = np.random.normal()
        self.w2 = np.random.normal()
        self.w3 = np.random.normal()
        self.w4 = np.random.normal()
        self.w5 = np.random.normal()
        self.w6 = np.random.normal()
        
        self.b1 = np.random.normal()
        self.b2 = np.random.normal()
        self.b3 = np.random.normal()
        
    def feedforward(self, x):
        h1 = activate(self.w1 * x[0] + self.w2 * x[1] + self.b1)
        h2 = activate(self.w3 * x[0] + self.w4 * x[1] + self.b2)
        o1 = activate(self.w5 * h1 + self.w6 * h2 + self.b3)
        return o1
    
    def train(self, inputs, dependant_vec):
        
        learn_rate = 0.1 
        epochs = 1000
        for epoch in range(epochs):
            for x,y in zip(inputs, dependant_vec):
                sum_h1 = self.w1 * x[0] + self.w2 * x[1] + self.b1
                h1 = activate(sum_h1)
        
                sum_h2 = self.w3 * x[0] + self.w4 * x[1] + self.b2
                h2 = activate(sum_h2)
        
                sum_o1 = self.w5 * h1 + self.w6 * h2 + self.b3
                o1 = activate(sum_o1)
                y_pred = o1
                
                d_L_d_ypred = -2 * (y - y_pred)

                # Neuron o1
                d_ypred_d_w5 = h1 * derivate_activation(sum_o1)
                d_ypred_d_w6 = h2 * derivate_activation(sum_o1)
                d_ypred_d_b3 = derivate_activation(sum_o1)
        
                d_ypred_d_h1 = self.w5 * derivate_activation(sum_o1)
                d_ypred_d_h2 = self.w6 * derivate_activation(sum_o1)
        
                # Neuron h1
                d_h1_d_w1 = x[0] * derivate_activation(sum_h1)
                d_h1_d_w2 = x[1] * derivate_activation(sum_h1)
                d_h1_d_b1 = derivate_activation(sum_h1)
        
                # Neuron h2
                d_h2_d_w3 = x[0] * derivate_activation(sum_h2)
                d_h2_d_w4 = x[1] * derivate_activation(sum_h2)
                d_h2_d_b2 = derivate_activation(sum_h2)
        
                # --- Update weights and biases
                # Neuron h1
                self.w1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w1
                self.w2 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w2
                self.b1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_b1
        
                # Neuron h2
                self.w3 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w3
                self.w4 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w4
                self.b2 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_b2
        
                # Neuron o1
                self.w5 -= learn_rate * d_L_d_ypred * d_ypred_d_w5
                self.w6 -= learn_rate * d_L_d_ypred * d_ypred_d_w6
                self.b3 -= learn_rate * d_L_d_ypred * d_ypred_d_b3
                
            if epoch % 5 == 0:
                y_pred_vec = np.apply_along_axis(self.feedforward, 1, inputs)
                loss = error_loss(dependant_vec, y_pred_vec)
                print("epoch %d loss: %.3f" % (epoch, loss))
                    
                    
                    

data = np.array([
  [-2, -1],  # Alice
  [25, 6],   # Bob
  [17, 4],   # Charlie
  [-15, -6], # Diana
])
output = np.array([
  1, # Alice
  0, # Bob
  0, # Charlie
  1, # Diana
])

# Train our neural network!
network = NeuralNet()
network.train(data, output)


# epoch 975 loss: 0.002
# epoch 980 loss: 0.002
# epoch 985 loss: 0.002
# epoch 990 loss: 0.002
# epoch 995 loss: 0.002
