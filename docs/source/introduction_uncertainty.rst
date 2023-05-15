Introduction to classification uncertainty and modern deep neural networks
==========================================================================

We are currently working on a tutorial and the improvement of this page.

Short background
----------------

The native uncertainty estimate built in neural classifiers is the softmax.
To recall, we normalize the pointwise exponential of the output of the classifier.
It follows that the outputs are included in \[0, 1\] and that their sum equals 1.

The neural network therefore outputs the vector of the parameters of a categorical distribution,
also called multinoulli, that represents the probability for the input to belongs to a class i.
Yet we can wonder whether we can trust this estimation.

The overconfidence of neural networks
-------------------------------------


References
----------
