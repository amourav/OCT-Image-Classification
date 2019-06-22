import os
from os.path import join
from keras.models import load_model


p = r"/scratch/amourav/Python/OCT-Image-Classification/modelOutput/metaClf_2019-06-15_/Xception/Xception.hdf5"

outD = os.path.dirname(p)
out = join(outD, "Xception_weights.hdf5")
print(p)
model = load_model(p)
model.summary()
model.save_weights(out)
