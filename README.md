# Improving-Sepsis-Treatment-Strategies
This is the code repository for paper "Improving Sepsis Treatment Strategies using Deep Reinforcement Learning and Mixture-of-Experts"

### Motivation
Sepsis is the leading cause of mortality in the ICU.  It is challenging to manage because different patients respond differently to treatment.  Thus, tailoring treatment to the individual patient is essential for the best outcomes.  In this paper, we take steps toward this goal by applying a mixture-of-experts framework to individualize sepsis treatment. The mixture model switches between neighbor-based (kernel) and deep reinforcement learning (DRL) experts depending on patient's current history.  On a large retrospective cohort, this mixture-based approach outperforms physician, kernel only, and DRL-only experts.

### Data Preprocessing

### Pipeline
The following diagram illustrate the whole pipeline. There are three major components, they are patient **encoder**, **expert policies derivation**, and **policy evaluation**.

<img src="/pipeline.png" width="500">


### Encoders
You can find the encoders under ```./src/encoder```

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
The Figure below illustrates the what kernel expert does. The nearest neighbors can be found based on either recurrent state representation or non-recurrent one.

#### DQN
#### MoE

### WDR estimator
