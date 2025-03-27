from models import Sequential
from layers import Dense
import numpy as np
import time

def large():
    model = Sequential()
    model.add(Dense(512, 512))
    model.add(Dense(256, 512))
    model.add(Dense(128, 256))
    model.add(Dense(64, 128))
    model.add(Dense(32, 64))
    model.add(Dense(32, 32, activation='softmax'))

    start = time.time()
    x = model.output(np.random.normal(size=(512,)))
    model.train(np.random.normal(size=(1,512,)), np.random.normal(size=(1,32,)), 1000)
    print(time.time() - start)

def and_nn():
    and_nn = Sequential(1)
    and_nn.add(Dense(1,2))
    and_nn.layers[0].weights = np.array([-.5, .5])
    and_nn.layers[0].biases = np.array([1.5])

    and_in = np.array([[0,0],[1,0],[0,1],[1,1]])
    and_t = np.array([[0],[0],[0],[1]])

    and_nn.train(and_in, and_t, 1000)
    print(and_nn.layers[0].weights, and_nn.layers[0].biases)

def xor():
    xor_nn = Sequential(1)
    xor_nn.add(Dense(2, 2))  # First hidden layer with 2 neurons
    xor_nn.add(Dense(1, 2))  # Output layer with 1 neuron

    # Initialize weights and biases
    xor_nn.layers[0].weights = np.array([[0.2, -0.4], [0.7, 0.1]])
    xor_nn.layers[0].biases = np.array([0.0, 0.0])
    xor_nn.layers[1].weights = np.array([[0.6, 0.9]])
    xor_nn.layers[1].biases = np.array([0.0])

    xor_in = np.array([[0,0],[1,0],[0,1],[1,1]])
    xor_t = np.array([[0],[1],[1],[0]])

    xor_nn.train(xor_in, xor_t, 1000)

    print(xor_nn.layers[0].weights, xor_nn.layers[0].biases)

def adder():
    adder_nn = Sequential(1)
    adder_nn.add(Dense(3, 2))  # Hidden layer with 3 neurons
    adder_nn.add(Dense(2, 3))  # Output layer with 2 neurons

    # Initialize weights and biases
    adder_nn.layers[0].weights = np.array([[0.0, 0.1], [0.2, 0.3], [0.4, 0.5]])
    adder_nn.layers[0].biases = np.array([0.0, 0.0, 0.0])
    adder_nn.layers[1].weights = np.array([[0.6, 0.7, 0.8], [0.9, 1.0, 1.1]])
    adder_nn.layers[1].biases = np.array([0.0, 0.0])

    adder_in = np.array([[0,0],[1,0],[0,1],[1,1]])
    adder_t = np.array([[0,0],[0,1],[0,1],[1,0]])

    adder_nn.train(adder_in, adder_t, 1000)

    print(adder_nn.layers[0].weights, adder_nn.layers[0].biases)

if __name__ == '__main__':
    large()
    and_nn()
    xor()
    adder()