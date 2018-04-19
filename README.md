# Improving-Sepsis-Treatment-Strategies
This is the code repository for paper "Improving Sepsis Treatment Strategies using Deep Reinforcement Learning and Mixture-of-Experts"

### Motivation
Sepsis is the leading cause of mortality in the ICU.  It is challenging to manage because different patients respond differently to treatment.  Thus, tailoring treatment to the individual patient is essential for the best outcomes.  In this paper, we take steps toward this goal by applying a mixture-of-experts framework to individualize sepsis treatment. The mixture model switches between neighbor-based (kernel) and deep reinforcement learning (DRL) experts depending on patient's current history.  On a large retrospective cohort, this mixture-based approach outperforms physician, kernel only, and DRL-only experts.

### Pipeline
The following diagram illustrate the whole pipeline. There are three major components, they are patient **encoder**, **expert policies derivation**, and **policy evaluation**. We will go through the code for each component.

