import numpy as np

#We start off by implementing a step function
def stepFunction(t):
    if(t>=0):
        return 1 
    else:
        return 0

#Performs the linear operation W*X+b. Passes the output to the 
#Step function 
def prediction(X, W, b):
    return stepFunction((np.matmul(X, W)+b))

#For the length of the input vector, we predict the value yhat
#Compare yhat with the actual value of y and update the weights 
#appropriately. 
#If it is a positive point in the  negative direction. We add to the 
#weights and bias. Else we substract from the weights and return the weights and bias.
def perceptron(X, W, y, b, learn_rate=0.01):
    for i in range(len(X)):
        yhat=prediction(X[i], W[i], b)
        if y[i]-yhat==1:
            W[0]+=X[i][0]*learn_rate
            W[1]+=X[i][1]*learn_rate
            b+=learn_rate
        elif y[i]-yhat==-1:
            W[0]-=X[i][0]*learn_rate
            W[1]-=X[i][1]*learn_rate
            b-=learn_rate
    return W, b
            

def trainPerceptron(X, y, learn_rate=0.01, num_epochs=25):
    x_min, x_max=min(X.T[0]), max(X.T[0])
    y_min, y_max=min(X.T[1]), max(X.T[1])
    
    W=np.array(np.random.rand(2,1))
    b=np.random.rand(1)[0]+x_max
    
    #These are the solution lines that get plotted
    boundary_lines=[]
    for i in range(num_epochs):
        #In each epoch we apply the perceptron step
        W, b=perceptron(X,y, W, b, learn_rate)
        boundary_lines.append((-W[0]/W[1], -b/W[1]))
        return boundary_lines