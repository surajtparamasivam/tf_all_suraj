#tf.estimator

import numpy as np
import tensorflow as tf

feature_columns=[tf.feature_column.numeric_column("x",shape=[1])]
estimator=tf.estimator.LinearRegressor(feature_columns=feature_columns)

x_train=np.array([1.,2.,3.,4.])
y_train=np.array([0.,-1.,-2.,-3])
x_eval = np.array([2., 5., 8., 1.])
y_eval = np.array([-1.01, -4.1, -7, 0.])

input_fn = tf.estimator.inputs.numpy_input_fn(
    {"x": x_train}, y_train, batch_size=4, num_epochs=None, shuffle=True)

input_train_fn=tf.estimator.inputs.numpy_input_fn({"x":x_train},y_train,batch_size=4,num_epochs=1000,shuffle=False)
input_eval_fn=tf.estimator.inputs.numpy_input_fn({"x":x_eval},y_eval,batch_size=4,num_epochs=1000,shuffle=False)

estimator.train(input_fn=input_train_fn,steps=1000)
train_metrics=estimator.evaluate(input_fn=input_train_fn)
eval_metrics=estimator.evaluate(input_fn=input_eval_fn)
print("train metrics %r"%train_metrics)
print("eval metrics %r"% eval_metrics)
