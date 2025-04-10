from models import Sequential
from layers import Dense
from tensor import Tensor
import mlx.core as mx
import time

def large(ap=0):
    model = Sequential()
    model.add(Dense(2**(6+ap), 2**(6+ap),learning_rate=model.learning_rate))
    model.add(Dense(2**(5+ap), 2**(6+ap),learning_rate=model.learning_rate))
    model.add(Dense(2**(4+ap), 2**(5+ap),learning_rate=model.learning_rate))
    model.add(Dense(2**(3+ap), 2**(4+ap),learning_rate=model.learning_rate))
    model.add(Dense(2**(2+ap), 2**(3+ap),learning_rate=model.learning_rate))
    model.add(Dense(2**(1+ap), 2**(2+ap), activation='softmax'))

    start = time.time()
    # x = model.output(mx.random.normal(shape=(2**(6+ap),)))
    model.train(Tensor(mx.random.normal(shape=(1,2**(6+ap),))), Tensor(mx.random.normal(shape=(1,2**(1+ap),))), 1000)
    return time.time() - start

def and_nn():
    and_nn = Sequential(1)
    and_nn.add(Dense(1,2,learning_rate=and_nn.learning_rate))
    and_nn.layers[0].weights = Tensor(mx.array([-.5, .5]))
    and_nn.layers[0].biases = Tensor(mx.array([1.5]))

    and_in = Tensor(mx.array([[0,0],[1,0],[0,1],[1,1]]))
    and_t = Tensor(mx.array([[0],[0],[0],[1]]))

    x = and_nn.output(Tensor(and_in.data[0], requires_grad=True))

    and_nn.train(and_in, and_t, 1000)
    print(and_nn.layers[0].weights, and_nn.layers[0].biases)

def xor():
    xor_nn = Sequential(1)
    xor_nn.add(Dense(2, 2, learning_rate=xor_nn.learning_rate))
    xor_nn.add(Dense(1, 2, learning_rate=xor_nn.learning_rate))

    # Initialize weights and biases
    xor_nn.layers[0].weights = Tensor(mx.array([[0.2, -0.4], [0.7, 0.1]]))
    xor_nn.layers[0].biases = Tensor(mx.array([0.0, 0.0]))
    xor_nn.layers[1].weights = Tensor(mx.array([[0.6, 0.9]]))
    xor_nn.layers[1].biases = Tensor(mx.array([0.0]))

    xor_in = Tensor(mx.array([[0,0],[1,0],[0,1],[1,1]]))
    xor_in = Tensor(mx.array([[1,1],[0,1],[1,0],[0,0]]))
    xor_t = Tensor(mx.array([[0],[1],[1],[0]]))

    xor_nn.output(Tensor(xor_in.data[0], requires_grad=True))

    xor_nn.train(xor_in, xor_t, 1000)
    print(xor_nn.layers[0].weights, xor_nn.layers[0].biases)

def adder():
    adder_nn = Sequential(1)
    adder_nn.add(Dense(3, 2, learning_rate=adder_nn.learning_rate))  # Hidden layer with 3 neurons
    adder_nn.add(Dense(2, 3, learning_rate=adder_nn.learning_rate))  # Output layer with 2 neurons

    # Initialize weights and biases
    adder_nn.layers[0].weights = Tensor(mx.array([[0.0, 0.1], [0.2, 0.3], [0.4, 0.5]]))
    adder_nn.layers[0].biases = Tensor(mx.array([0.0, 0.0, 0.0]))
    adder_nn.layers[1].weights = Tensor(mx.array([[0.6, 0.7, 0.8], [0.9, 1.0, 1.1]]))
    adder_nn.layers[1].biases = Tensor(mx.array([0.0, 0.0]))

    adder_in = Tensor(mx.array([[0,0],[1,0],[0,1],[1,1]]))
    adder_t = Tensor(mx.array([[0,0],[0,1],[0,1],[1,0]]))

    adder_nn.output(Tensor(adder_in.data[0]))

    adder_nn.train(adder_in, adder_t, 1000)
    print(adder_nn.layers[0].weights, adder_nn.layers[0].biases)


if __name__ == '__main__':
    # times = []
    # for ap in range(10):
    #     times.append(sum([large(ap) for _ in range(10)])/10)
    #     print(f'ap: {ap} Done')
    # print(times)
    and_nn()
    xor()
    adder()