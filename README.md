# Landscape image classification

A comparison of a convolutional neural network and decision-tree classifier for six landscape-image categories.

Training and evaluation are separate scripts. The CNN uses convolution/pooling/dropout layers; the tree baseline uses flattened image pixels. Evaluation scripts produce classification reports, confusion matrices and timing information.

Install `requirements.txt` and supply the image dataset under the configured `data/seg_train` and `data/seg_test` directories. Keep a separate held-out test set: the CNN's configured validation set is used by early stopping. Check model-save paths before running evaluation.

Dataset images and trained model artifacts are not included. University machine-learning study by Ovidiu Catalin Ghibea.
