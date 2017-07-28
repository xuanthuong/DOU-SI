# ref: https://machinelearningcoban.com/2017/01/01/kmeans/

import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame


columns = ["Key", "x", "y"]
test_data = [ ["Shipper", 2, 2], ["Consignee", 8, 3], ["Notify", 3, 6],
              ["Shipper", 2, 3], ["Consignee", 8, 4], ["Notify", 5, 6] ]
df = DataFrame(test_data, columns=columns)

def SI_keys_display(df):
  
  plt.figure(figsize=(4, 6))

  plt.plot(df['x'][df['Key'] == 'Shipper'], df['y'][df['Key'] == 'Shipper'], 'b^', markersize = 4, alpha = .8)
  plt.plot(df['x'][df['Key'] == 'Consignee'], df['y'][df['Key'] == 'Consignee'], 'go', markersize = 4, alpha = .8)
  plt.plot(df['x'][df['Key'] == 'Notify'], df['y'][df['Key'] == 'Notify'], 'rs', markersize = 4, alpha = .8)

  # plt.axis((0, 10, 0, 15))
  # plt.xlim(0, 10)
  # plt.ylim(0, 15)
  # plt.autoscale(False)
  # plt.axis('equal')
  
  plt.plot()
  plt.show()
    
SI_keys_display(df)