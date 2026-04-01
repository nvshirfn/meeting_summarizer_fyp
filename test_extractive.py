import inspect
import scipy
import numpy as np

# Apply Patches
if not hasattr(inspect, 'getargspec'):
    inspect.getargspec = inspect.getfullargspec
if not hasattr(scipy, 'asarray'):
    scipy.asarray = np.asarray
if not hasattr(scipy, 'ones'):
    scipy.ones = np.ones
if not hasattr(scipy, 'zeros'):
    scipy.zeros = np.zeros
if not hasattr(scipy, 'array'):
    scipy.array = np.array

import malaya
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

text = "Ini adalah ayat pertama. Ini pula ayat kedua. Ayat ketiga sangat panjang dan mengandungi banyak maklumat. Ayat keempat pendek. Ayat kelima mencuba nasib. Mari kita lihat ayat keenam."

print("Testing LSA...")
svd = TruncatedSVD(n_components = 2)
vectorizer = TfidfVectorizer()
model = malaya.summarization.extractive.sklearn(svd, vectorizer)
result = model.sentence_level(text, top_k=3)
print(result)

print("Done.")
