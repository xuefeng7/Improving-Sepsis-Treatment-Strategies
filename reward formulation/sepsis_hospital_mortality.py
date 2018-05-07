import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import matplotlib.gridspec as gridspec
from dataset import *

def generate_dataset(datadir='./data/sepsis'):
	features = np.load(datadir+'/x.npy')
	rewards = np.load(datadir+'/reward.npy')
	patients = np.load(datadir+'/patient_uid.npy')
	feature_names = open(datadir+'/x_colnames.txt', 'r').read().split("\n")[:-1]
	reward_names = open(datadir+'/reward_colnames.txt', 'r').read().split("\n")[:-1]
	return features, rewards, feature_names, reward_names, patients

class SepsisHospitalMortality(Dataset):
	def __init__(self, validation_size=2500, rwd=0, datadir='./data/sepsis'):
		# grab the dataset from the directory. By default, data from different patients is
		# sequentially grouped but all in the same array. `uids` gives the patient id.
		features, rewards, feature_names, reward_names, uids = generate_dataset(datadir=datadir)

		# Get the indexes of our training, validation, and test data
		filename = datadir + '/sepsis-dataset-indices.npz'.format(rwd)
		if os.path.exists(filename):
			cache = np.load(filename)
			idx_train, idx_val, idx_test, idx_rest = tuple([cache[f] for f in sorted(cache.files)])
		else:
			# Get all the indexes where mortality is true, and the first chunk
			# of equal size where mortality is false.
			idx_1 = np.argwhere(rewards[:, rwd] == 1)[:,0]
			idx_0 = np.argwhere(rewards[:, rwd] == 0)[:,0][:len(idx_1)]
			idx_0_rest = np.argwhere(rewards[:, rwd] == 0)[:,0][len(idx_1):]

			# determine all the indexes where the patient ID changes.
			split_points = (np.argwhere(uids[:,0][1:] - uids[:,0][:-1]) + 1)[:,0]

			# We want about 80% train, 20% test, but we don't want train and test
			# to contain data for the same patients, so find indexes near 80% where the uid
			# changes.
			n = int(len(idx_0) * 0.8)
			n0 = next(i for i in range(n-5, n+20) if idx_0[i] in split_points)
			n1 = next(i for i in range(n-5, n+20) if idx_1[i] in split_points)

			# Now we have train and test indexes into the original array of features/labels
			idx_0_train = idx_0[:n0]
			idx_1_train = idx_1[:n1]
			idx_0_test = idx_0[n0:]
			idx_1_test = idx_1[n1:]

			# Let's further split our training set into actual train and validation
			v = validation_size
			nn0 = next(i for i in range(v-5, v+20) if idx_0_train[i] in split_points)
			nn1 = next(i for i in range(v-5, v+20) if idx_1_train[i] in split_points)

			# Do the same thing as we did earlier
			idx_0_val = idx_0_train[:nn0]
			idx_1_val = idx_1_train[:nn1]
			idx_0_train = idx_0_train[nn0:]
			idx_1_train = idx_1_train[nn1:]

			# combine the indices
			idx_train = np.hstack((idx_0_train, idx_1_train))
			idx_val = np.hstack((idx_0_val, idx_1_val))
			idx_test = np.hstack((idx_0_test, idx_1_test))
			idx_rest = idx_0_rest

			# shuffle them so patient data is no longer sequential
			np.random.shuffle(idx_train)
			np.random.shuffle(idx_val)
			np.random.shuffle(idx_test)
			np.random.shuffle(idx_rest)

			# save them
			data = (idx_train, idx_val, idx_test, idx_rest)
			np.savez(filename, *data)

		self.features = features
		self.rewards = rewards[:, rwd]
		self.idx_train = idx_train
		self.idx_val = idx_val
		self.idx_test = idx_test
		self.idx_rest = idx_rest
		self.feature_names = feature_names
		self.label_names = ['Survived', 'Died in Hospital']

	@property
	def X(self): return self.features[self.idx_train]
	@property
	def y(self): return self.rewards[self.idx_train]
	@property
	def Xt(self): return self.features[self.idx_test]
	@property
	def yt(self): return self.rewards[self.idx_test]
	@property
	def Xv(self): return self.features[self.idx_val]
	@property
	def yv(self): return self.rewards[self.idx_val]
	@property
	def Xr(self): return self.features[self.idx_rest]
	@property
	def yr(self): return self.rewards[self.idx_rest]

	def visualize_prediction_and_explanation(self, x, y, prob, grad):
		fig = plt.figure(figsize=(20, 8))
		G = gridspec.GridSpec(6, 10)

		mag = np.abs(grad).max()
		pred = int(round(prob))
		norm = Normalize(vmin=-1, vmax=1)
		sm = ScalarMappable(norm=norm, cmap=plt.cm.bwr)
		xmin = self.Xv.min(axis=0)
		xmax = self.Xv.max(axis=0)
		xmed = (xmin + xmax) * 0.5

		plt.subplot(G[:,:2])
		self.explanation_barchart(grad)
		plt.title('$\hat{y}='+str(pred)+'$')
		plt.gca().set_yticklabels(['{}: {:.2f}'.format(n, x[j]) for j,n in enumerate(self.feature_names)], fontsize=9)

		for i, label in enumerate(sorted(self.feature_names)):
			divisor = 1.0
			if label == 'age': divisor = 365.0
			j = self.feature_names.index(label)
			weight = (grad[j]/mag)**2 * np.sign(grad[j])
			plt.subplot(G[i//8, 2+i%8], axisbg=sm.to_rgba(weight))
			plt.hist(self.Xv[:,j]/divisor, bins=25, alpha=0.5, color='blue')
			plt.gca().set_yticklabels([])
			plt.axvline(x[j]/divisor, ls='--', lw=2, color='black')
			plt.tick_params(axis='both', which='major', labelsize=8)
			plt.xticks(np.array([xmin[j], xmed[j], xmax[j]])/divisor)
			plt.title(label[:16] + ": {:.1f}".format(x[j]/divisor), fontsize=8)

		fig.suptitle('Prediction = {} ({:.1%}), True Outcome = {}'.format(
				self.label_names[pred], prob,
				self.label_names[y]), fontsize=16)
		plt.tight_layout(rect=[0, 0.03, 1, 0.95])
		plt.show()
			
if __name__ == '__main__':
	# dataset = SepsisHospitalMortality()
	# #import pdb; pdb.set_trace()
	# model = dataset.twolayer_mlp()
	# model.fit(features, onehot(rewards))
	pass
