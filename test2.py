import os
from os.path import join
from keras.models import model_from_json
from keras.models import model_from_yaml


weightsPath = r"/scratch/amourav/Python/OCT-Image-Classification/modelOutput/metaClf_2019-06-22_/Xception/Xception.hdf5"
jsonPath = r"/scratch/amourav/Python/OCT-Image-Classification/modelOutput/metaClf_2019-06-22_/Xception/Xception_architecture.json"
yamlPath = r"/scratch/amourav/Python/OCT-Image-Classification/modelOutput/metaClf_2019-06-22_/Xception/Xception_architecture.yaml"

print(weightsPath)

# load json and create model
print('load json')
json_file = open(jsonPath, 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights(weightsPath)
print("Loaded json model from disk")
loaded_model.summary()

print('load yaml')
yaml_file = open(yamlPath, 'r')
loaded_model_yaml = yaml_file.read()
yaml_file.close()
loaded_model = model_from_yaml(loaded_model_yaml)
# load weights into new model
loaded_model.load_weights(weightsPath)
print("Loaded yaml model from disk")
loaded_model.summary()

