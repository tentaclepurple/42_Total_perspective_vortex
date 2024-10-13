#!/usr/bin/env python
# coding: utf-8

# In[5]:


ls


# In[4]:


import os
import mne
import shutil
import requests
import pandas as pd


# In[10]:


#os.chdir("42_Total_perspective_vortex")

if not os.path.exists('data'):
    os.makedirs('data')


# In[12]:


# URL del archivo a descargar
url = "https://physionet.org/files/eegmmidb/1.0.0/S001/S001R01.edf?download"
local_filename = "data/S001R01.edf"

response = requests.get(url, stream=True)

if response.status_code == 200:
    with open(local_filename, 'wb') as file:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                file.write(chunk)
    print(f"Archivo descargado correctamente como {local_filename}")
else:
    print(f"Error en la descarga. Status code: {response.status_code}")


# In[14]:


path = "data/S001R01.edf"
raw_data = mne.io.read_raw_edf(path, preload=True)
print("RAW", raw_data)
raw_data.plot()


# In[ ]:




