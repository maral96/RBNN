import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import seaborn as sn
import matplotlib.pyplot as plt

data = np.load('features.npy')
target = np.load('target.npy')
# Picking the top 1000 points as TSNE takes a lot of time for 15K points
data = data[:1000,]
labels = target[:1000,]
model = TSNE(n_components=2, random_state=0)
# configuring the parameteres
# the number of components = 2
# default perplexity = 30
# default learning rate = 200
# default Maximum number of iterations for the optimization = 1000
tsne_data = model.fit_transform(data)
# creating a new data frame which help us in ploting the result data
tsne_data = np.vstack((tsne_data.T, labels)).T
tsne_df = pd.DataFrame(data=tsne_data, columns=('Dim_1', 'Dim_2', 'label'))
# Ploting the result of tsne
sn.FacetGrid(tsne_df, hue='label', size=6).map(plt.scatter, 'Dim_1', 'Dim_2').add_legend()
plt.savefig('./tsne.png')