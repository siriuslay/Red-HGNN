import pickle

from scipy import sparse
import numpy as np


# features_0 = sparse.load_npz('../imdb_raw/features_0.npz')
# feature0 = features_0.A
# features_1 = sparse.load_npz('../imdb_raw/features_1.npz')
# feature1 = features_1.A
# features_2 = sparse.load_npz('../imdb_raw/features_2.npz')
# feature2 = features_2.A
# feat = np.vstack((feature0, feature1, feature2))
# features = sparse.csr_matrix(feat)
# with open("../imdb/node_features.pkl", "wb") as tf:
#     pickle.dump(features, tf)

#
# old_labels = np.load('../imdb_raw/labels.npy')
# train_val_test_idx = np.load('../imdb_raw/train_val_test_idx.npz')
# train_idx = train_val_test_idx['train_idx']
# val_idx = train_val_test_idx['val_idx']
# test_idx = train_val_test_idx['test_idx']
#
# la0 = np.zeros((400, 2), dtype=int)
# la0[:, 0] = train_idx[:]
# la0[:, 1] = old_labels[train_idx]
#
# la1 = np.zeros((400, 2), dtype=int)
# la1[:, 0] = val_idx[:]
# la1[:, 1] = old_labels[val_idx]
#
# la2 = np.zeros((3478, 2), dtype=int)
# la2[:, 0] = test_idx[:]
# la2[:, 1] = old_labels[test_idx]
# labels = []
# labels.append(la0)
# labels.append(la1)
# labels.append(la2)

# with open("../imdb/label.txt", "w") as f:
#     for i in range(len(old_labels)):
#         f.write(f'm{i} {old_labels[i]}\n')
#     f.close()
# for i in range(len(old_labels))

# with open("../imdb/labels.pkl", "wb") as tf:
#     pickle.dump(labels, tf)


adjM = sparse.load_npz('../imdb_raw/adjM.npz')
all_adj = adjM.A

md = all_adj
md[:, 0:4278] = 0
md[:, 6359:11616] = 0
md[4278:, :] = 0
MD = sparse.csr_matrix(md)
ma = adjM.A
ma[:, 0:6359] = 0
ma[4278:, :] = 0
MA = sparse.csr_matrix(ma)

edges = {'m-d': MD, 'm-a': MA}

with open("../imdb/edge2.txt", "w") as f:
    for i in range(11616):
        for j in range(11616):
            if ma[i,j] != 0:
                f.write(f'{i}\t{j}\tm-a\t1\n')
                f.write(f'{j}\t{i}\ta-m\t1\n')


# with open("../imdb/edges.pkl", "wb") as tf:
#     pickle.dump(edges, tf)

# type_mask = np.load('../imdb_raw/node_types.npy')