from __future__ import print_function
from __future__ import absolute_import
import six
import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
from utils import *
from neural_network import *

class Dataset(object):
  @cachedproperty
  def num_features(self):
    return self.X.shape[1]

  @cachedproperty
  def num_classes(self):
    return len(np.unique(self.y))

  @cachedproperty
  def onehot_y(self):
    y = onehot(self.y)
    assert(y.shape[1] == self.onehot_yt.shape[1])
    return y

  @cachedproperty
  def onehot_yt(self):
    return onehot(self.yt)

  @cachedproperty
  def onehot_yv(self):
    return onehot(self.yv)

  @property
  def Xf(self):
    if hasattr(self, 'Xv'):
      return np.vstack((self.X, self.Xv))
    else:
      return self.X

  @property
  def yf(self):
    if hasattr(self, 'Xv'):
      return np.hstack((self.y, self.yv))
    else:
      return self.y

  @property
  def onehot_yf(self):
    return onehot(self.yf)

  def explanation_barchart(self, coefs):
    colors = ['orange' if c > 0 else 'blue' for c in coefs]
    data = list(zip(coefs, self.feature_names))
    coefs = [d[0] for d in data]
    labels = [d[1] for d in data]
    plt.barh(range(len(labels)), coefs, color=colors, align='center')
    plt.ylim(-0.5, len(labels))
    ax = plt.gca()
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticks([min(coefs), max(coefs)])
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xticklabels(self.label_names, fontsize=9)
    ax.grid(axis='y')

  @cachedproperty
  def naive_forest_score(self):
    return Score(self.yt, self.naive_forest.predict_proba(self.Xt))

  @cachedproperty
  def naive_logreg_score(self):
    return Score(self.yt, self.naive_logreg.predict_proba(self.Xt))

  @cachedproperty
  def onelayer_mlp(self):
    return self.define_network(hidden_layers=[50], dropout_at=[0])

  @cachedproperty
  def twolayer_mlp(self):
    return self.define_network(hidden_layers=[50,30], dropout_at=[1])

  @cachedproperty
  def threelayer_mlp(self):
    return self.define_network(hidden_layers=[50, 40, 30], dropout_at=[2])

  def define_network(self, hidden_layers=[50,30], dropout_at=[1], keep_prob=0.8):
    import tensorflow as tf
    import tensorflow.contrib.slim as slim

    name = 'FullyConnected{}'.format('x'.join(map(str,hidden_layers)))

    network = type(
        self.__class__.__name__ + name, (NeuralNetwork,), {
          'x_shape': lambda _: [None, self.num_features],
          'y_shape': lambda _: [None, self.num_classes] })

    network.x_shape = property(network.x_shape)
    network.y_shape = property(network.y_shape)

    def rebuild_model(o, X, reuse=None):
      sizes = [self.num_features] + list(hidden_layers) + [self.num_classes]
      layers = [X]
      for i, activation in enumerate([tf.nn.relu for _ in hidden_layers] + [None]):
        out = layers[i]
        if i-1 in dropout_at:
          out = slim.dropout(out, keep_prob, is_training=o.is_train)
        out = slim.fully_connected(
            out, sizes[i+1], reuse=reuse,
            scope=o.name+'/L{}'.format(i),
            activation_fn=activation)
        layers.append(out)
      return layers

    network.rebuild_model = six.create_unbound_method(rebuild_model, network)

    return network
