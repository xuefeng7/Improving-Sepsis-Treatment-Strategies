from __future__ import absolute_import
from __future__ import print_function
import uuid
import time
import os
import numpy as np
import tensorflow as tf
import six
import six.moves.cPickle as pickle
from six import add_metaclass
from abc import ABCMeta, abstractmethod, abstractproperty
from utils import *

"""
Class attempting to make Tensorflow models more object-oriented
and similar to sklearn's fit/predict interface.
"""
@add_metaclass(ABCMeta)
class NeuralNetwork():
  def __init__(self, name=None, dtype=tf.float32, **kwargs):
    self.vals = None
    self.name = (name or str(uuid.uuid4()))
    self.dtype = dtype
    self.setup_model(**kwargs)
    assert(hasattr(self, 'X'))
    assert(hasattr(self, 'y'))
    assert(hasattr(self, 'logits'))

  def setup_model(self, X=None, y=None):
    with tf.name_scope(self.name):
      self.X = tf.placeholder(self.dtype, self.x_shape, name="X") if X is None else X
      self.y = tf.placeholder(self.dtype, self.y_shape, name="y") if y is None else y
      self.is_train = tf.placeholder_with_default(
          tf.constant(False, dtype=tf.bool), shape=(), name="is_train")
    self.model = self.rebuild_model(self.X)
    self.recompute_vars()

  @property
  def logits(self):
    return self.model[-1]

  def rebuild_model(self, X, reuse=None):
    """Define all of your Tensorflow variables here, making sure to scope them
    under `self.name`, and also making sure to return a list/tuple whose final element
    is your network's logits. In subclasses, remember to call super!"""

  @abstractproperty
  def x_shape(self):
    """Specify the shape of X; for MNIST, this could be [None, 784]"""

  @abstractproperty
  def y_shape(self):
    """Specify the shape of y; for MNIST, this would be [None, 10]"""

  @property
  def num_features(self):
    """Helper to return the dimensionality of X (aka D)"""
    return np.product(self.x_shape[1:])

  @property
  def num_classes(self):
    """Helper to return the dimensionality of y (aka K)"""
    return np.product(self.y_shape[1:])

  ###################################################
  # Useful (cached) functions of our network outputs
  #
  @property
  def trainable_vars(self):
    return [v for v in tf.trainable_variables() if v in self.vars]

  def input_grad(self, f):
    return tf.gradients(f, self.X)[0]

  def param_grad(self, f):
    return tf.gradients(f, self.trainable_vars)

  def cross_entropy_with(self, y):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=y))

  @cachedproperty
  def preds(self):
    """Symbolic TF variable returning an Nx1 vector of predictions"""
    return tf.argmax(self.logits, axis=1)

  @cachedproperty
  def probs(self):
    """Symbolic TF variable returning an NxK vector of probabilities"""
    return tf.nn.softmax(self.logits)

  @cachedproperty
  def logps(self):
    """Symbolic TF variable returning an NxK vector of log-probabilities"""
    return self.logits - tf.reduce_logsumexp(self.logits, 1, keep_dims=True)

  @cachedproperty
  def l1_weights(self):
    """L1 loss for the weights of the network"""
    return tf.add_n([tf.reduce_sum(tf.abs(v)) for v in self.trainable_vars])

  @cachedproperty
  def l2_weights(self):
    """L2 loss for the weights of the network"""
    return tf.add_n([tf.nn.l2_loss(v) for v in self.trainable_vars])

  @cachedproperty
  def cross_entropy(self):
    """Symbolic TF variable returning information distance between the model's
    predictions and the true labels y"""
    return self.cross_entropy_with(self.y)

  @cachedproperty
  def cross_entropy_input_gradients(self):
    return self.input_grad(self.cross_entropy)

  @cachedproperty
  def cross_entropy_param_gradients(self):
    return self.param_grad(self.cross_entropy)

  @cachedproperty
  def logprob_sum_input_gradients(self):
    return self.input_grad(self.logps)

  @cachedproperty
  def logprob_sum_param_gradients(self):
    return self.param_grad(self.logps)

  @cachedproperty
  def l1_double_backprop(self):
    return l1_loss(self.cross_entropy_input_gradients)

  @cachedproperty
  def l2_double_backprop(self):
    return l2_loss(self.cross_entropy_input_gradients)

  @cachedproperty
  def binary_logits(self):
    assert(self.num_classes == 2)
    return self.logps[:,1] - self.logps[:,0]

  @cachedproperty
  def binary_logit_input_gradients(self):
    return self.input_grad(self.binary_logits)

  @cachedproperty
  def accuracy(self):
    return tf.reduce_mean(tf.cast(tf.equal(self.preds, tf.argmax(self.y, 1)), dtype=tf.float32))

  #############################################
  # Predicting
  #
  def score(self, X, y, **kw):
    """Function that takes numpy arrays `X` (NxD) and `y` (Nx1)
    and returns the model's predictive accuracy"""
    if len(y.shape) > 1: y = np.argmax(y, axis=1)
    return np.mean(y == self.predict(X, **kw))

  def predict(self, X, **kw):
    """Function that takes numpy arrays `X` (in NxD) and returns Nx1 predictions (batched)"""
    with tf.Session() as sess:
      self.init(sess)
      return self.predict_(sess, X, **kw)

  def predict_logits(self, X, **kw):
    with tf.Session() as sess:
      self.init(sess)
      return self.logits_(sess, X, **kw)

  def predict_binary_logodds(self, X, **kw):
    with tf.Session() as sess:
      self.init(sess)
      return self.binary_logits_(sess, X, **kw)

  def predict_proba(self, X, **kw):
    """Function that takes numpy arrays `X` (in NxD) and returns NxK predicted
    probabilities"""
    with tf.Session() as sess:
      self.init(sess)
      return self.probs_(sess, X, **kw)

  # private functions that implement the above methods without starting a new
  # Tensorflow session
  def probs_(self, sess, X, n=256):
    probs = sess.run(self.probs, feed_dict={ self.X: X[:n] })
    for i in range(n, len(X), n):
      probs = np.vstack((probs, sess.run(self.probs, feed_dict={ self.X: X[i:i+n] })))
    return probs

  def predict_(self, sess, X, n=256):
    preds = sess.run(self.preds, feed_dict={ self.X: X[:n] })
    for i in range(n, len(X), n):
      preds = np.hstack((preds, sess.run(self.preds, feed_dict={ self.X: X[i:i+n] })))
    return preds

  def logits_(self, sess, X, n=256):
    logits = sess.run(self.logits, feed_dict={ self.X: X[:n] })
    for i in range(n, len(X), n):
      logits = np.vstack((logits, sess.run(self.logits, feed_dict={ self.X: X[i:i+n] })))
    return logits

  def binary_logits_(self, sess, X, n=256):
    logits = sess.run(self.binary_logits, feed_dict={ self.X: X[:n] })
    for i in range(n, len(X), n):
      logits = np.hstack((logits, sess.run(self.binary_logits, feed_dict={ self.X: X[i:i+n] })))
    return logits

  ###################################################
  # Explaining
  #
  def input_gradients(self, X, y=None, n=256, **kw):
    """Batched version of input gradients"""
    with tf.Session() as sess:
      self.init(sess)
      return self.input_gradients_(sess, X, y, n, **kw)

  def input_gradients_(self, sess, X, y=None, n=256, **kw):
    yy = y[:n] if y is not None and not isint(y) else y
    grads = self.compute_input_gradients_(sess, X[:n], yy, **kw)
    for i in range(n, len(X), n):
      yy = y[i:i+n] if y is not None and not isint(y) else y
      grads = np.vstack((grads,
        self.compute_input_gradients_(sess, X[i:i+n], yy, **kw)))
    return grads

  def compute_input_gradients_(self, sess, X, y=None, logits=False):
    if y is None:
      return sess.run(self.logprob_sum_input_gradients, feed_dict={ self.X: X })
    elif logits and self.num_classes == 2:
      return sess.run(self.binary_logit_input_gradients, feed_dict={ self.X: X })
    elif isint(y):
      y = onehot(np.array([y]*len(X)), self.num_classes)
    return sess.run(self.cross_entropy_input_gradients, feed_dict={ self.X: X, self.y: y })

  #############################################
  # Training
  #
  def minibatches(self, kwargs, batch_size=128, num_epochs=32):
    """Helper to generate minibatches of the training set (called by `fit`).
    Currently this just iterates sequentially through `kwargs['X']` for
    `num_epochs`, taking `batch_size` examples per iteration. If you need
    fancier behavior, you can override this function or provide your own batch
    generator to pass to `fit_batches`. """
    assert('X' in kwargs or self.X in kwargs)
    X = kwargs.get('X', kwargs.get(self.X, None))
    n = int(np.ceil(len(X) / batch_size))
    tensors = self.parse_placeholders(kwargs)
    for i in range(int(num_epochs * n)):
      idx = slice((i%n)*batch_size, ((i%n)+1)*batch_size)
      feed = {}
      for var, value in six.iteritems(tensors):
        feed[var] = value[idx]
      yield feed

  def loss_function(self, l1_weights=0., l1_double_backprop=0.,
                          l2_weights=0., l2_double_backprop=0.):
    """By default, still just use the cross entropy as the loss, but allow
    users to penalize the L1 or L2 norm of the input sensitivity by passing
    the given params below to `fit`."""
    log_likelihood = self.cross_entropy
    log_prior = 0
    for reg in ['l1_double_backprop', 'l1_weights',
                'l2_double_backprop', 'l2_weights']:
      if eval(reg) > 0:
        log_prior += eval(reg) * getattr(self, reg)
    return log_likelihood + log_prior

  def fit(self, X, y, Xv, yv, l1, batch_size=128, num_epochs=50, **kwargs):
    """Trains the model for the specified duration. See `minibatches` and
    `fit_batches` for option definitions"""
    # batch_kw = {}
    # batch_kw.update(kwargs)
    # batch_kw['X'] = X
    # batch_kw['y'] = y
    #data = self.minibatches(batch_kw, batch_size=batch_size, num_epochs=num_epochs)
    # num_batches = X.shape // batch_size
    # for batch_idx in range(num_batches):
    return self.fit_batches(X, y, Xv, yv, batch_size, num_epochs, loss_fn=self.loss_function(l1_double_backprop=l1), **kwargs)

  def fit_batches(self, X, y, Xv, yv, batch_size, num_epochs,
      optimizer=None, loss_fn=None, print_every=100, val_every=5,
      init=False, initial_learning_rate=0.1, lr_decay_interval=10, lr_decay_factor=0.5,
      epsilon=1e-8,
      call_every=None, callback=None, capper=None, **kwargs):
    """
    Actually fit the model using the `batches` iterator, which should yield
    successive feed_dicts containing new examples. This is designed to be
    flexible so you can either iterate through a giant array in memory or
    have a queue loading files and doing preprocessing (though if you're
    really doing serious training, you're probably better off making all
    of the preprocessing symbolic and writing a clunky megascript).

    You can pass a custom optimizer, loss function, gradient capping function,
    or callback. You can also choose to reinitialize the model with its current param
    values before starting.
    """
    learning_rate = tf.placeholder(tf.float32, shape=[])
    if optimizer is None:
      # optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=epsilon)
      optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    if loss_fn is None:
      loss_fn = self.loss_function(**kwargs)

    grads_and_vars = optimizer.compute_gradients(loss_fn)

    if capper is not None:
      grads_and_vars = capper(grads_and_vars)

    train_op = optimizer.apply_gradients(grads_and_vars)
    accuracy = self.accuracy
    t = time.time()

    losses, accs = [], []
    losses_v, accs_v = [], []

    def learning_rate_for(epoch):
      lr = initial_learning_rate
      while epoch > lr_decay_interval:
        lr *= lr_decay_factor
        epoch -= lr_decay_interval
      return lr

    with tf.Session() as sess:
      # Init
      sess.run(tf.global_variables_initializer())
      if init: self.init(sess)

      # Train
      num_batches = X.shape[0] // batch_size
      for epoch in range(num_epochs):
        for batch_idx in range(num_batches):
          batch_X, batch_y = X[batch_idx*batch_size:(batch_idx+1)*batch_size], y[batch_idx*batch_size:(batch_idx+1)*batch_size]
          _, loss, accu = sess.run([train_op, loss_fn, accuracy], feed_dict={
              self.is_train: True,
              self.X: batch_X,
              self.y: batch_y,
              learning_rate: learning_rate_for(epoch)
            })
          # batch_loss += loss
          # batch_acc += accu
          # if print_every and i % print_every == 0:
        #   #   print('Batch {}, loss {:.4f}, accuracy {:.4f}, {:.1f}s'.format(batch_idx, loss, accu, time.time() - t))
        losses += [loss]
        accs += [accu]

        print('Epoch {}, loss {:.4f}, accuracy {:.4f}, {:.1f}s'.format(epoch + 1, losses[-1], accs[-1], time.time() - t))
        
        if epoch + 1 == num_epochs: #and epoch % val_every == 0:
          val_loss, val_accu = sess.run([loss_fn, accuracy], feed_dict={
              self.is_train: False,
              self.X: Xv,
              self.y: yv
            })
          print('Val: epoch {}, loss {:.4f}, accuracy {:.4f}'.format(epoch + 1, val_loss, val_accu))
      # for i, batch in enumerate(batches):
      #   batch[self.is_train] = True
      #   _, loss, accu = sess.run([train_op, loss_fn, accuracy], feed_dict=batch)
      #   losses += [loss]
      #   accs += [accu]
      #   if print_every and i % print_every == 0:
      #     print('Batch {}, loss {:.4f}, accuracy {:.4f}, {:.1f}s'.format(i, loss, accu, time.time() - t))
      #   if call_every and i % call_every == 0:
      #     callback(sess, self, batch, i)

      # # Save
      self.vals = [v.eval() for v in self.vars]
      return losses, accs, val_loss, val_accu

  #############################################
  # Persisting/loading variables
  #
  def recompute_vars(self):
    """Find all of the (trainable) variables that belong to this model"""
    self.vars = tf.get_default_graph().get_collection(
        tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)

  def init(self, sess):
    """Assign all of the stored `vals` to our Tensorflow `vars`. Run this after
    you start a new session."""
    if self.vals is None:
      sess.run(tf.global_variables_initializer())
    else:
      for var, val in zip(self.vars, self.vals):
        sess.run(var.assign(val))

  def save(self, filename):
    """Save our variables as lists of numpy arrays, because they're a bit easier to work
    with than giant tensorflow directories."""
    with open(filename, 'wb') as f:
      pickle.dump(self.vals, f)

  def load(self, filename):
    """Load saved variables"""
    with open(filename, 'rb') as f:
      self.vals = pickle.load(f)

  def parse_placeholders(self, kwargs):
    """
    Figure out which elements of a dictionary are either tf.placeholders
    or strings referencing attributes that are tf.placeholders, then ensure
    we populate the feed dict with actual placeholders for easy feeding later.
    """
    feed = {}
    for dictionary in [kwargs, kwargs.get('feed_dict', {})]:
      for key, val in six.iteritems(dictionary):
        attr = getattr(self, key) if isinstance(key, str) and hasattr(self, key) else key
        if type(attr) == type(self.X):
          if len(attr.shape) > 1:
            if attr.shape[0].value is None:
              feed[attr] = val
    return feed
