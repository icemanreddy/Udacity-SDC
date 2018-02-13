from keras.models import load_model
from keras.models import model_from_json
import os
import sys

# python load_model.py <json> <weights> <model_name>
json_file = open(sys.argv[1], 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights(sys.argv[2])
print("Loaded model from disk")
loaded_model.summary()
loaded_model.save(sys.argv[3])

