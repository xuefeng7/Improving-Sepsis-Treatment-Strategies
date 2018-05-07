import numpy as np
import tensorflow as tf

def isint(x):
  return isinstance(x, (int, np.int32, np.int64))

def onehot(Y, K=None):
  if K is None:
    K = np.unique(Y)
  elif isint(K):
    K = list(range(K))
  data = np.array([[y == k for k in K] for y in Y]).astype(int)
  return data

def l1_loss(x):
  return tf.reduce_sum(tf.abs(x))

def l2_loss(x):
  return tf.nn.l2_loss(x)

class cachedproperty(object):
  def __init__(self, function):
    self.__doc__ = getattr(function, '__doc__')
    self.function = function

  def __get__(self, instance, klass):
    if instance is None: return self
    value = instance.__dict__[self.function.__name__] = self.function(instance)
    return value
