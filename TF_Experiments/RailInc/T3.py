import tensorflow as tf
import tensorflow_hub as hub

import numpy as np
import os
import pandas as pd

dataset=pd.read_csv(r'trip_event_explanation.csv',header=1,error_bad_lines=False)
# print(dataset.columns)
