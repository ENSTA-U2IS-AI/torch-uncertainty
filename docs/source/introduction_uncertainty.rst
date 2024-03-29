:orphan:

Introduction to Classification Uncertainty
==========================================

WORK IN PROGRESS - Please wait while we are currently working on a tutorial and the improvement of this page...

Short Background
----------------

Neural networks in classification settings have a built-in uncertainty estimation
based on the last layer activation function: the softmax.

To recall, the softmax normalizes the outputs of the neural networks (also called logits), taking its pointwise exponential
and normalizing the sum of the outputs to 1. It follows that the final values are included in \[0, 1\] and that their sum equals 1.

The neural network therefore outputs the vector of the parameters of a categorical distribution,
also called multinoulli, that represents the probability for the input to belongs to the class corresponding to its index.

However, the softmax is not a calibrated estimator of the uncertainty. Conceptually, it is not the true probability, but the likelihood
given the trained model. This likelihood is known to be overconfident, and depending on the use case,
it may not be a good idea to trust these predictions. Let's see why in more details.

The Overconfidence of Neural Networks
-------------------------------------

References
----------

- Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. . On calibration of modern neural networks. In ICML 2017.
