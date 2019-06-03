import numpy as np 
import tensorflow as tf 
import matplotlib.pyplot as plt 
np.random.seed(101) 
tf.set_random_seed(101) 
# Genrating random linear data 
# There will be 50 data points ranging from 0 to 50 
x = np.linspace(0, 50, 50) 
# Adding noise to the random linear data 
x += np.random.uniform(-4, 4, 50)
'''
line_1:

Now generate the value of Y randomly from standard normal distribution. 
Make sure the shape of X and Y are same

'''
y=np.linspace(-2,2,50)
y+=tp.random.normal(-4,4,50)

n = len(x) # Number of data points 
# Plot of Training Data 
plt.scatter(x, y) 
plt.xlabel('x') 
plt.ylabel('y') 
plt.title("Training Data") 
plt.show() 

X = tf.placeholder("float") 
Y = tf.placeholder("float") 
W = tf.Variable(np.random.randn(), name = "W") 
b = tf.Variable(np.random.randn(), name = "b")
'''
line_2 & line_3 : 
Now , create two variables named learning_rate and training_epochs and set some value. 
First assign learning rate as 0.01 and training epochs as 1000

line_4:

declare the hypothesis line as 

y_pred= X*W + b

use tensorflow (tf) to add and mutiply 

line_5:

Declare the cost function as mean squared error of y_pred and Y as

         1                         2 
cost =  --- [ sum of ( y_pred - Y )   ]
         2n
         
Use tf.reduce_sum and tf.pow 

'''
learning_rate = 0.01
training_epochs = 1000
y_pred = tf.add(tf.multiply(X,W),b)
cost=tf.reduce_sum(tf.pow(y_pred-Y,2))/(2*n)


# Gradient Descent Optimizer 
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost) 

# Global Variables Initializer 
init = tf.global_variables_initializer() 
# Starting the Tensorflow Session 
with tf.Session() as sess: 
    sess.run(init) 
    # Iterating through all the epochs 
    for epoch in range(training_epochs): 
        # Feeding each data point into the optimizer using Feed Dictionary 
        for (_x, _y) in zip(x, y): 
            sess.run(optimizer, feed_dict = {X : _x, Y : _y}) 
        # Displaying the result after every 50 epochs 
        if (epoch + 1) % 50 == 0: 
            # Calculating the cost a every epoch 
            c = sess.run(cost, feed_dict = {X : x, Y : y}) 
            print("Epoch", (epoch + 1), ": cost =", c, "W =", sess.run(W), "b =", sess.run(b)) 

    # Storing necessary values to be used outside the Session 
    training_cost = sess.run(cost, feed_dict ={X: x, Y: y}) 
    weight = sess.run(W) 
    bias = sess.run(b) 

# Calculating the predictions 
predictions = weight * x + bias 
print("Training cost =", training_cost, "Weight =", weight, "bias =", bias, '\n')  

# Plotting the Results 
plt.plot(x, y, 'ro', label ='Original data') 
plt.plot(x, predictions, label ='Fitted line') 
plt.title('Linear Regression Result') 
plt.legend() 
plt.show() 
