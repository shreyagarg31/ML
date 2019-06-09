import numpy as np
X=np.random.randint(2,size=(3,4))
Y=np.random.randint(2,size=(3,1))

def sigmoid(x):
    return 1/(1+np.exp(-x))
def derivative_sigmoid(x):
    return x*(1-x)
weights=2*np.random.random((4,1)) - 1
neurons=4
epochs=10000
for iteration in range(epochs):
    input_layer=X
    outputs=sigmoid(np.dot(input_layer,weights))
    error=Y - outputs
    adjust=error*derivative_sigmoid(outputs)
    weights+=np.dot(input_layer.T,adjust)

print("Outputs are:")
print(outputs)

print("Actual values were:")
print(Y)























