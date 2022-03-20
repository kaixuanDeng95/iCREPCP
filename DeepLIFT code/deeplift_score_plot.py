# %%
import numpy as np
from deeplift.visualization import viz_sequence

X = np.load('./filename.npy')

for idx in range(X.shape[0]):
    print("Scores for example {}:".format(idx))
    scores_for_idx = X[idx]
    viz_sequence.plot_weights(scores_for_idx, subticks_frequency=10,figsize=(100,10))


# %%
