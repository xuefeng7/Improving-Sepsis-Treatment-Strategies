# Improving-Sepsis-Treatment-Strategies
This is the code repository for paper "Improving Sepsis Treatment Strategies using Deep Reinforcement Learning and Mixture-of-Experts"

### Motivation
Sepsis is the leading cause of mortality in the ICU.  It is challenging to manage because different patients respond differently to treatment.  Thus, tailoring treatment to the individual patient is essential for the best outcomes.  In this paper, we take steps toward this goal by applying a mixture-of-experts framework to individualize sepsis treatment. The mixture model switches between neighbor-based (kernel) and deep reinforcement learning (DRL) experts depending on patient's current history.  On a large retrospective cohort, this mixture-based approach outperforms physician, kernel only, and DRL-only experts.

### Data Preprocessing

### Pipeline
The following diagram illustrate the whole pipeline. There are three major components, they are patient **encoder**, **expert policies derivation**, and **policy evaluation**.

<img src="/pipeline.png" width="500">


### Encoders
You can find the encoders under ```./src/encoder```, both encoders are written in *Pytorch*, you can follow the [instruction](http://pytorch.org) to install it.

#### Recurrent autoencoder
The recurrent autoencoder regards the entire clinical course for a patient as a sample so that each sample contains multiple patient observations. In our case, for each observation, there are 45 features, and the encoder maps them into a state representation with length 128, and the decoder reconstruct the original features for each observation based on this representation. For training, the reconstruction loss is measured by MSE, and the metric used is R-square. To train it with your own samples, just call

```{python}
autoencoder = autoEncoder(45, 128)
train_autoencoder(train_set, test_set, autoencoder, num_epoch=50, val=True)
```

Then, to obtain the state representations for each observation, call

```{python}
train_embeddings = do_eval(train_set, autoencoder, output_embeddings=True)
test_embeddings = do_eval(test_set, autoencoder, output_embeddings=True)
```

#### Sparse autoencoder
Unlike the recurrent autoencoder, sparse autoencoder does not consider the temporal correlations of the observations within a patient's clinical course. Instead, it treats each observation, regardless which patient it belongs to, as a single sample. In our case, it maps the 45 features into a 128 state representation with sparse constrain, empirically it will make the policy learning easier. The code for running sparse autoencoder is similar to running the recurrent autoencoder shown above. Check the ```.ipynb``` for more details.

### Experts

In our study, there are three experts used for policy learning. Knernel is a neighor-based policy learning expert, DQN is the model-free policy learning expert, and MoE is a mix of kernel and DQN.  

#### Kernel
The Figure below illustrates the what kernel expert does. 

<img src="/kernel.png" width="500">

The circle in the left shows an example of the neighborhoods of a new state ```s```, red and green marks the mortality and surviving states respectively, and each of these states is associated with a physician action ```A_i```

The nearest neighbors can be found based on either recurrent state representation or non-recurrent one. The kernel expert is implemented in ```./src/expert/kernel.ipynb```

Note that here kernel has two tasks. 1) Deriving kernel policy; 2) Estimate original physician policies. For task 1), kernel expert finds nearest neighboring states (from trainset) for each state in the trainset. For state in testset, kernel also finds nearest neighboring states (from trainset) for it. For task 2), kernel finds nearest neighboring states (from trainset) for each state in the trainset; however, please be careful that kernel finds nearest neighboring states (from testset) for each state in the testset. More details can be found in the ```.ipynb```.


#### DQN

##### Intermediate Rewards(IR)

To stably train the DQN experts and improve the policy learning quality, we formulate a way to compute IR. More specificaly, we use the change in mortality probabilities of state transistion as the IR. To obtain it, run the code in ```./src/expert/rewards/IR.ipynb```, this ```.ipynb``` depends on the files under ```./src/expert/rewards/nn```, you can even customize the network architecture by modifying those files accordingly.

##### DQN expert training

The DQN expert is a Dueling Double Deep Q-network. It is written in ```./src/expert/qnetwork.py``` using *Tensorflow*. To run it, command 

```
python qnetwork.py 
  --num_steps 200000 
  --input_train path/to/trainset 
  --input_test path/to/testset 
  --output_train_meta path/to/store/losses/and/mean/Q/values
  --output_trainset path/to/store/training/results(q-values, actions)
  --output_testset path/to/store/testing/results(q-values, actions)
```

Please note that the input train/test sets should have the following format, you can generate them via the code in ```generate_dqn_dataset.ipynb```

```
dqn_df = pd.DataFrame(embeddings) # all state representations (sample_size, 128)
dqn_df['icustayid'] = ICU stay id for each patient
dqn_df['vaso_input'] = discrete action index for vaso usage
dqn_df['iv_input'] = discrete action index for iv usage
dqn_df['reward'] = in our case, negative log-odds mortality prob.
```

We solve the *V* and *Q* for physician policies by slightly modify the qnetwork. The DQN's Q-function is learned during training by taking **argmax** . Since the kernel and physician expert policies are not determined via RL, they do not have associated *V* and *Q* functions, and must therefore be estimated in order to calculate the WDR (see last section). We used the DQN's Q-function, using the **mean**  operation rather than **argmax** in order to derive the equivalent physician policy.

The physician policies Q can be obtained in the same fashion. Just run

```
python qnetwork_solve_phy_VQ.py 
  --num_steps 200000 
  --input_train path/to/trainset 
  --input_test path/to/testset 
  --output_train_meta path/to/store/losses/and/mean/Q/values
  --output_trainset path/to/store/training/results(q-values, actions)
  --output_testset path/to/store/testing/results(q-values, actions)
```

using the same input train/test sets. Then call, 

```
dqn_res_train = pkl.load(open(path/to/store/training/results,'rb')) # a tuple (Q, actions)
phy_train_Q = dqn_res_train[0] # (sample_size, num_actions)
phy_test_V = phy_train_Q.max(axis = 1)
```

#### MoE

MoE combines the policies from kernel and DQN organically. It is trained to select expert for treatment per patient per state based on current physiological condition (characteristics) of the patient.

<img src="/moe.png" width="300">

The MoE is trained to optimize the WDR estimates (see last section) for the discount expected return. The MoE is implemented in ```./src/expert/MoE.ipynb``` using *Pytorch*. More details can be found in the jupyter notebook.

### WDR estimator

The weighted doubly robust (WDR) estimator is widely used for off-policy evaluation in RL. You can find its implementation in ```./src/expert/MoE.ipynb```. WDR is a very complicated function with many local maxima. We optimize it through large amount of simulations with random initial points, but the global maxima is still not guaranteed. 
