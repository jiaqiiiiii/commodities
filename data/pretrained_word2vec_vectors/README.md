## Pretrained Word2Vec Vectors

The static embedding analysis uses pretrained and aligned Word2Vec vectors from:

Nilo Pedrazzini, & Barbara McGillivray. (2022). Diachronic word embeddings from 19th-century newspapers digitised by the British Library (1800-1919) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.7181682

These vectors were trained on the same HMD and LwM data with the following hyperparameters: skip-gram architecture, 5 epochs, 200 dimensions, context window of 3, minimum word count of 1. Decade-specific vector spaces were aligned using Orthogonal Procrustes (Schönemann, 1966).

We do not redistribute these vectors. Please download these vectors from Zenodo: https://zenodo.org/records/7181682. 
