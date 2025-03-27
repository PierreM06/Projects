from models import Sequential
from layers import Dense
import mlx.core as mx
import time

def large(ap=0):
    model = Sequential()
    model.add(Dense(2**(6+ap), 2**(6+ap)))
    model.add(Dense(2**(5+ap), 2**(6+ap)))
    model.add(Dense(2**(4+ap), 2**(5+ap)))
    model.add(Dense(2**(3+ap), 2**(4+ap)))
    model.add(Dense(2**(2+ap), 2**(3+ap)))
    model.add(Dense(2**(1+ap), 2**(2+ap), activation='softmax'))

    start = time.time()
    # x = model.output(mx.random.normal(shape=(2**(6+ap),)))
    model.train(mx.random.normal(shape=(1,2**(6+ap),)), mx.random.normal(shape=(1,2**(1+ap),)), 1000)
    return time.time() - start

def and_nn():
    and_nn = Sequential(1)
    and_nn.add(Dense(1,2))
    and_nn.layers[0].weights = mx.array([-.5, .5])
    and_nn.layers[0].biases = mx.array([1.5])

    and_in = mx.array([[0,0],[1,0],[0,1],[1,1]])
    and_t = mx.array([[0],[0],[0],[1]])

    and_nn.train(and_in, and_t, 1000)
    print(and_nn.layers[0].weights, and_nn.layers[0].biases)

def xor():
    xor_nn = Sequential(1)
    xor_nn.add(Dense(2, 2))
    xor_nn.add(Dense(1, 2))

    # Initialize weights and biases
    xor_nn.layers[0].weights = mx.array([[0.2, -0.4], [0.7, 0.1]])
    xor_nn.layers[0].biases = mx.array([0.0, 0.0])
    xor_nn.layers[1].weights = mx.array([[0.6, 0.9]])
    xor_nn.layers[1].biases = mx.array([0.0])

    xor_in = mx.array([[0,0],[1,0],[0,1],[1,1]])
    xor_in = mx.array([[1,1],[0,1],[1,0],[0,0]])
    xor_t = mx.array([[0],[1],[1],[0]])

    xor_nn.train(xor_in, xor_t, 1000)

    print(xor_nn.layers[0].weights, xor_nn.layers[0].biases)

def adder():
    adder_nn = Sequential(1)
    adder_nn.add(Dense(3, 2))  # Hidden layer with 3 neurons
    adder_nn.add(Dense(2, 3))  # Output layer with 2 neurons

    # Initialize weights and biases
    adder_nn.layers[0].weights = mx.array([[0.0, 0.1], [0.2, 0.3], [0.4, 0.5]])
    adder_nn.layers[0].biases = mx.array([0.0, 0.0, 0.0])
    adder_nn.layers[1].weights = mx.array([[0.6, 0.7, 0.8], [0.9, 1.0, 1.1]])
    adder_nn.layers[1].biases = mx.array([0.0, 0.0])

    adder_in = mx.array([[0,0],[1,0],[0,1],[1,1]])
    adder_t = mx.array([[0,0],[0,1],[0,1],[1,0]])

    adder_nn.train(adder_in, adder_t, 1000)

    print(adder_nn.layers[0].weights, adder_nn.layers[0].biases)


if __name__ == '__main__':
    times = []
    for ap in range(10):
        times.append(sum([large(ap) for _ in range(10)])/10)
        print(f'ap: {ap} Done')
    print(times)
    and_nn()
    xor()
    adder()