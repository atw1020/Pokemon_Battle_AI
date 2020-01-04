'''

Author: Arthur Wesley

'''

import itertools
import pickle

from tensorflow import keras
import numpy as np
import Parser
import Replay_Web_Scrapper
import training_data_generator

print(training_data_generator.get_validation_accuracy("gen7ou"))
