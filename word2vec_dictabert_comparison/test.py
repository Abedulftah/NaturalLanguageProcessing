import os
import numpy as np
from gensim.models import Word2Vec

model = Word2Vec.load(os.path.join("C:/Users/User/Desktop/knesset_word2vec.model"))

