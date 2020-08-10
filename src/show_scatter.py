import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
from DataManager import DataManager
from util import TRAIN, SMALL

model = torch.load('.//data//budala_6.pickle', map_location=torch.device('cpu'))

#data_manager_train = DataManager()

#tag_to_idx, idx_to_tag = data_manager_train.get_tag_dicts()

embedding = model.module.tag_embeddings.weight

embedding = TSNE(n_components=2, ).fit_transform(embedding.detach().long().numpy())
embedding_with = np.array([ [x,y,i] for i, (x, y) in enumerate(embedding)], dtype=np.int32)
#plt.scatter(embedding[:, 0], embedding[:, 1])


x = embedding[:, 0]
y = embedding[:, 1]
names = np.array(list("ABCDEFGHIJKLMNO"))


c = np.random.randint(1,5,size=15)

fig, ax = plt.subplots()
sc = plt.scatter(x,y)

annot = ax.annotate("", xy=(0,0), xytext=(20,20),textcoords="offset points",
                    bbox=dict(boxstyle="round", fc="w"),
                    arrowprops=dict(arrowstyle="->"))
annot.set_visible(False)

def update_annot(ind):

    pos = sc.get_offsets()[ind["ind"][0]]
    annot.xy = pos
    text = "{}".format(" ".join(list(map(str,ind["ind"]))), 
                           " ".join([str(embedding_with[n,2]) for n in ind["ind"]]))
    annot.set_text(text)
    annot.get_bbox_patch().set_alpha(0.4)


def hover(event):
    vis = annot.get_visible()
    if event.inaxes == ax:
        cont, ind = sc.contains(event)
        if cont:
            update_annot(ind)
            annot.set_visible(True)
            fig.canvas.draw_idle()
        else:
            if vis:
                annot.set_visible(False)
                fig.canvas.draw_idle()

fig.canvas.mpl_connect("motion_notify_event", hover)

plt.show()

#print(embedding)