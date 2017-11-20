{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<strong> Recurrent Neural Networks </strong> <br/>\n",
    "A recurrent neural network can be thought of as multiple copies of the same network, each passing a message to a successor. \n",
    "\n",
    "![](img/RNN.png)\n",
    "\n",
    "\n",
    "All recurrent neural networks have the form of a chain of repeating modules of neural network. In standard RNNs, this repeating module will have a very simple structure, such as a single tanh layer.\n",
    "\n",
    "![](img/RNN-unrolled.png)\n",
    "\n",
    "h(t) = tanh(W[x(t), h(t-1)] + b)<br/><br/><br/>\n",
    "However, in practice, RNNs were shown to be unable to handle Long-term dependencies.\n",
    "\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "collapsed": true
   },
   "source": [
    "<strong> LSTM </strong> <br/>\n",
    "Long Short Term Memory networks – usually just called “LSTMs” – are a special kind of RNN, capable of learning long-term dependencies. \n",
    "\n",
    "LSTMs also have the same chain like structure, but the repeating module has a different structure. Instead of having a single neural network layer, there are four, interacting in a very special way.\n",
    "![](img/LSTM-chain.png)\n",
    "\n",
    "The key to LSTMs is the cell state. It is like a conveyor belt running straight down the entire chain, with only some minor linear interactions. It’s very easy for information to just flow along it unchanged.\n",
    "![](img/C.png)\n",
    "\n",
    "![](img/LSTM1.png)\n",
    "![](img/LSTM2.png)\n",
    "![](img/LSTM3.png)\n",
    "![](img/LSTM4.png)\n",
    "\n",
    "\n",
    "<b>Reference</b>: Olah, C., 2015. Understanding lstm networks. GITHUB blog, posted on August, 27, p.2015."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<strong> Model implementation in TensorFlow </strong> <br/>\n",
    "\n",
    "We create a network that has only one LSTM cell. We pass 2 elemnts to LSTM, the h(t-1) and c(t-1) which are called <b> state</b>. Here, state is a tuple with 2 elements, each one is of size [1 x 4], one for passing prv_output to next time step, and another for passing the prv_state to next time stamp."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import tensorflow as tf\n",
    "from tensorflow.contrib import rnn\n",
    "sess = tf.Session()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "---\n",
    "### Let's Understand the parameters, inputs and outputs\n",
    "\n",
    "We will treat the MNIST image $\\in \\mathcal{R}^{28 \\times 28}$ as $28$ sequences of a vector $\\mathbf{x} \\in \\mathcal{R}^{28}$. \n",
    "\n",
    "![](img/mnist.png)\n",
    "\n",
    "<b>Reference</b>: Jasdeep Singh Chhabra, 2017. Understanding LSTM in Tensorflow(MNIST dataset). GITHUB blog."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "# Import data\n",
    "from tensorflow.examples.tutorials.mnist import input_data\n",
    "mnist = input_data.read_data_sets('MNIST_data', one_hot=True)\n",
    "trainX = mnist.train.images\n",
    "trainY = mnist.train.labels\n",
    "valX = mnist.validation.images\n",
    "valY = mnist.validation.labels\n",
    "testX = mnist.test.images\n",
    "testY = mnist.test.labels\n",
    "\n",
    "trainX = trainX.reshape(-1, 28, 28)\n",
    "testX = testX.reshape(-1, 28, 28)\n",
    "\n",
    "num_input = 28 # MNIST data input (img shape: 28*28)\n",
    "timesteps = 28 # timesteps\n",
    "num_lstm = 128 # hidden layer num of features\n",
    "num_classes = 10 # MNIST total classes (0-9 digits)\n",
    "batch_size = 128"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Placeholders"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "# tf Graph input\n",
    "X = tf.placeholder(\"float\", [None, timesteps, num_input])\n",
    "Y = tf.placeholder(\"float\", [None, num_classes])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "# Define weights and biases\n",
    "weights = {'out': tf.Variable(tf.random_normal([num_lstm, num_classes]))}\n",
    "biases = {'out': tf.Variable(tf.random_normal([num_classes]))}"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Constructing a basic <a href: \"https://www.tensorflow.org/api_docs/python/tf/contrib/rnn/LSTMCell\" >LSTM </a> cell with tensorflow"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "# Define the lstm cells\n",
    "lstm_cell = rnn.BasicLSTMCell(num_lstm, forget_bias=1.0)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "A simplest form of RNN in TensorFlow specified from lstm_cell: rnn.static_rnn(cells, inputs)\n",
    "<br/> The input argument has to be sequential (list of tensors) where the length of the list is the number of time steps.  "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "# Unstack to get a list of 'timesteps' tensors of shape (batch_size, n_input)\n",
    "x = tf.unstack(X, timesteps, 1)\n",
    "# Get lstm cell output\n",
    "outputs, states = rnn.static_rnn(lstm_cell, x,dtype=tf.float32)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The output generated by the static_rnn is a list of tensors of shape [batch_size,num_units]. <br/>\n",
    "Here, the output of the final step would be considered for the goal of classification.\n",
    "<br/>\n",
    "The states are tuple where the first element in the tuple is the cell state and the second is the hidden state.  "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "# Linear activation, using rnn inner loop last output\n",
    "logits = tf.matmul(outputs[-1], weights['out']) + biases['out']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "# Training Parameters\n",
    "learning_rate = 0.001\n",
    "numEpochs = 1000\n",
    "batch_size = 128\n",
    "display_step = 200\n",
    "avg_cost_val=[]\n",
    "train_accuracy=[]\n",
    "val_accuracy = []"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))\n",
    "optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)\n",
    "train_step = optimizer.minimize(cost)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "# Evaluate model (with test logits, for dropout to be disabled)\n",
    "output_layer = tf.nn.softmax(logits)\n",
    "prediction = tf.equal(tf.argmax(output_layer, 1), tf.argmax(Y, 1))\n",
    "accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "# Initialize the variables (i.e. assign their default value)\n",
    "init = tf.global_variables_initializer()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "# Start training\n",
    "with tf.Session() as sess:\n",
    "\n",
    "    # Run the initializer\n",
    "    sess.run(init)\n",
    "    for epoch in range(numEpochs):\n",
    "\n",
    "        print epoch\n",
    "        avg_cost = 0.\n",
    "        tr_avg_acc=0.\n",
    "        total_batch = int(mnist.train.num_examples/batch_size)\n",
    "        \n",
    "        _current_cell_state = np.zeros((batch_size, num_lstm))\n",
    "        _current_hidden_state = np.zeros((batch_size, num_lstm))\n",
    "        \n",
    "        # Loop over all batches\n",
    "        for _ in range(total_batch):\n",
    "            batch_x, batch_y = mnist.train.next_batch(batch_size)\n",
    "            # Reshape data to get 28 seq of 28 elements\n",
    "            batch_x = batch_x.reshape((batch_size, timesteps, num_input))\n",
    "            \n",
    "            # Run optimization op (backprop)\n",
    "            \n",
    "            _, c,tr_acc,current_states = sess.run([train_step,cost,accuracy,states], \n",
    "                                                  feed_dict={X: batch_x, Y: batch_y})\n",
    "            avg_cost += c / total_batch\n",
    "            tr_avg_acc += tr_acc/total_batch\n",
    "            \n",
    "            _current_cell_state,_current_hidden_state = current_states\n",
    "            \n",
    "        # Display logs per epoch step\n",
    "        avg_cost_val.append(avg_cost)\n",
    "        train_accuracy.append(tr_avg_acc)\n",
    "        # accuracy on validation set\n",
    "        valX = valX.reshape((len(valX), timesteps, num_input))\n",
    "        val_accuracy.append(sess.run(accuracy, feed_dict={X:valX,Y:valY}))\n",
    "        print(\"Epoch:\", '%04d' % (epoch+1), \"cost=\", \"{:.9f}\".format(avg_cost))\n",
    "            \n",
    "    \n",
    "    print(\"Optimization Finished :=) !\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "#Plot the cost, training and validation error\n",
    "import matplotlib.pyplot as plt\n",
    "plt.figure()\n",
    "plt.plot(avg_cost_val,'r-',label = 'cost')\n",
    "plt.figure()\n",
    "plt.plot(train_accuracy,'k*-')            \n",
    "plt.plot(val_accuracy,'bo-')\n",
    "plt.legend(('training accuracy', 'validation accuracy'), shadow=True, fancybox=True)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "# accuracy on test set\n",
    "testX = testX.reshape((len(testX), timesteps, num_input))\n",
    "print(sess.run(accuracy, feed_dict={x:testX,y:testY}))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "anaconda-cloud": {},
  "kernelspec": {
   "display_name": "Python [conda root]",
   "language": "python",
   "name": "conda-root-py"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 2
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython2",
   "version": "2.7.12"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 1
}
