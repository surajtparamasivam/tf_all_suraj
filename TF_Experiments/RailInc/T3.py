import tensorflow as tf
from matplotlib import pyplot
import pandas as pd
from datetime import datetime
dataset=pd.read_csv(r'carload/000000_0',header=0)
dataset=dataset.iloc[1:2]
x=dataset['trip_event_dtl_new_5lanes.end_event_ts'].map(lambda x: datetime.strptime(str(x), '%Y-%m-%d %H:%M:%S'))
y=dataset['trip_event_dtl_new_5lanes.start_event_ts'].map(lambda x: datetime.strptime(str(x), '%Y-%m-%d %H:%M:%S'))
z=dataset['trip_event_dtl_new_5lanes.start_location_splc_ext_nbr']
print(x)
print(y)
print(z)