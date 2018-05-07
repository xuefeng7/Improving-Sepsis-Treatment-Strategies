from sepsis_hospital_mortality import *
import matplotlib.pyplot as plt

dataset = SepsisHospitalMortality()
model = dataset.twolayer_mlp()
model.load('./sepsis-mortality-prediction-50x30-mlp-l1-gradients.pkl')

print('Train accuracy')
print(model.score(dataset.X, dataset.y))

print('Test accuracy')
print(model.score(dataset.Xt, dataset.yt))

print('Mortality probabilities')
predicted_mortality_probs = model.predict_proba(dataset.Xt)[:,1]
mort_true = np.argwhere(dataset.yt == 1)[:,0]
mort_false = np.argwhere(dataset.yt == 0)[:,0]
bins = np.linspace(0,1,20)
plt.hist(
    predicted_mortality_probs[mort_true],
    alpha=0.5, label='Died in hospital',
    color='red', bins=bins)
plt.hist(
    predicted_mortality_probs[mort_false],
    alpha=0.5, label='Survived',
    color='green', bins=bins)
plt.ylabel('# Test Examples')
plt.xlabel('Predicted Mortality Probability')
plt.xlim(0,1)
plt.legend(loc='best')
plt.title('Predicted mortality probabilities vs true outcomes')
plt.show()

print('Mortality log-odds')
predicted_mortality_logodds = model.predict_binary_logodds(dataset.Xt)
bins = np.linspace(-4, 4, 20)
plt.hist(
    predicted_mortality_logodds[mort_true],
    alpha=0.5, label='Died in hospital',
    color='red', bins=bins)
plt.hist(
    predicted_mortality_logodds[mort_false],
    alpha=0.5, label='Survived',
    color='green', bins=bins)
plt.ylabel('# Test Examples')
plt.xlabel('Predicted Mortality Log-odds')
plt.legend(loc='best')
plt.title('Predicted mortality log-odds vs true outcomes')
plt.show()

print('Prediction + explanation')
predicted_mortality_grads = model.input_gradients(dataset.Xt, y=1)
dataset.visualize_prediction_and_explanation(
    dataset.Xt[0],
    dataset.yt[0],
    predicted_mortality_probs[0],
    predicted_mortality_grads[0])
