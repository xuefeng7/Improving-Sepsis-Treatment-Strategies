from datasets import *
from sepsis import *
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import matplotlib.gridspec as gridspec
import os

model = 'lpt75-sparse-sepsis-mlp4'
if not os.path.exists(model):
  os.makedirs(model)

ds = SepsisHospitalMortality()
print(ds)

SepsisMLP = ds.create_tf_mlp(hidden_layers=[50,30], dropout_at=[1])
mlp = SepsisMLP()
mlp.load(model+'.pkl')
print(mlp.score(ds.Xt, ds.onehot_yt))

preds = mlp.predict(ds.Xv)
probs = mlp.predict_proba(ds.Xv)
grads = mlp.input_gradients(ds.Xv, y=np.array([[1,0] for _ in ds.Xv]))

xmin = ds.X_orig.min(axis=0)
xmax = ds.X_orig.max(axis=0)
xmed = (xmin + xmax) * 0.5

def present(idx, filename=None):
  fig = plt.figure(figsize=(20, 8))
  G = gridspec.GridSpec(6, 10)

  x = ds.Xv_orig[idx]
  grad = grads[idx]
  pred = preds[idx]
  prob = probs[idx][pred]
  mag = np.abs(grad).max()
  norm = Normalize(vmin=-1, vmax=1)
  sm = ScalarMappable(norm=norm, cmap=plt.cm.bwr)

  plt.subplot(G[:,:2])
  ds.explanation_barchart(grad)
  plt.title('$\hat{y}='+str(pred)+'$')
  plt.gca().set_yticklabels(['{}: {:.2f}'.format(n, x[j]) for j,n in enumerate(ds.feature_names)], fontsize=9)

  for i, label in enumerate(sorted(ds.feature_names)):
    divisor = 1.0
    if label == 'age': divisor = 365.0
    j = ds.feature_names.index(label)
    weight = (grad[j]/mag)**2 * np.sign(grad[j])
    plt.subplot(G[i//8, 2+i%8], axisbg=sm.to_rgba(weight))
    plt.hist(ds.X_orig[:,j]/divisor, bins=25, alpha=0.5, color='blue')
    plt.gca().set_yticklabels([])
    plt.axvline(x[j]/divisor, ls='--', lw=2, color='black')
    plt.tick_params(axis='both', which='major', labelsize=8)
    plt.xticks(np.array([xmin[j], xmed[j], xmax[j]])/divisor)
    plt.title(label[:16] + ": {:.1f}".format(x[j]/divisor), fontsize=8)

  fig.suptitle('Patient Index = {}, Prediction = {} ({:.1%}), True Outcome = {}'.format(
      idx,
      ds.label_names[pred],
      prob,
      ds.label_names[ds.yv[idx]]), fontsize=16)
  plt.tight_layout(rect=[0, 0.03, 1, 0.95])
  if filename:
    plt.savefig(filename)
    plt.close(fig)
  else:
    plt.show()

for i in range(len(ds.Xv)):
  present(i, model+'/{0:0>5}.png'.format(i))
