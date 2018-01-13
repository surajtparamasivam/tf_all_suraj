import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
plt.style.use('ggplot')

n_observations=1000
xs=np.linspace(-3,3,n_observations)
ys=np.sin(xs)+np.random.uniform(-0.5,0.5,n_observations)

# plt.scatter(xs,ys,alpha=0.15,marker='+')

X=tf.placeholder(tf.float32,name='X')
Y=tf.placeholder(tf.float32,name='Y')

sess=tf.InteractiveSession()
n=tf.random_normal([1000],stddev=0.1).eval()
W=tf.Variable(tf.random_normal([1],dtype=tf.float32,stddev=0.1),name='W')
B=tf.Variable(tf.constant([0],dtype=tf.float32),name='bias')
Y_pred=X*W+B

def distance(p1,p2):
    return tf.abs(p1-p2)

def train(X,Y,Y_pred,n_iterations=100,batch_size=200,learning_rate=0.02):
    cost=tf.reduce_mean(distance(Y_pred,Y))
    optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)


    n_iterations=500
    batch_size=1000
    fig,ax=plt.subplots(1,1)
    ax.scatter(xs,ys,marker='+',alpha=0.15)
    ax.set_xlim([-4,4])
    ax.set_ylim([-2,2])
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        prev_training_cost=0.0
        for it_i in range(n_iterations):
            sess.run(optimizer,feed_dict={X:xs,Y:ys})
            training_cost=sess.run(cost,feed_dict={X:xs,Y:ys})
            if it_i%10==0:
                ys_pred=Y_pred.eval(feed_dict={X:xs},session=sess)
                ax.plot(xs,ys_pred,'k',alpha=it_i/n_iterations)
                fig.show()
                plt.draw()
                print(training_cost)
            if np.abs(prev_training_cost-training_cost)< 0.000000001:
                 break

            prev_training_cost=training_cost
           
    n_neurons=100
    W=tf.Variable(tf.random_normal([1,n_neurons],dtype=tf.float32,stddev=0.1))
    b=tf.Variable(tf.constant(0,dtype=tf.float32,shape=[n_neurons]))
    h=tf.matmul(tf.expand_dims(X,1),W)+b
    Y_pred=tf.reduce_sum(h,1)
    train(X,Y,Y_pred)
    plt.show()