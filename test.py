import numpy as np
import pickle
f = np.load('data/trimmed_glove.npz')
a = f['embeddings']
print(a[0])
print(a[1])
print(a[2])
with open('data/word2index.pkl', 'rb') as f:
    a = pickle.load(f)
    print(a)


'''a = [1, 2, 3]
pickle_file = open('a.pkl', 'wb')
pickle.dump(a, pickle_file)
with open('a.pkl', 'rb') as f:
    a = pickle.load(f)
print(a[1])'''

