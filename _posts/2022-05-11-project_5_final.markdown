---
layout: post
title:      "Project 5 final"
date:       2022-05-11 17:39:43 +0000
permalink:  project_5_final
---

Student name: Zhiqiang Sun

Student pace: self paced

## Business understanding
The skin cancer dataset contains many medical images that show various kinds of skin cancer. In this project, we will analyze and visualize the relationship between cancer and age and the location of the body. Furthermore, we will use machine learning to train a model that can distinguish the cancer type by given images. 

## Dataset
The whole dataset were download from kaggle (https://www.kaggle.com/code/rakshitacharya/skin-cancer-data/data). The folder contains several csv files and two images folder. All the name of images were named with image id which can be found in the metadata excel file. There are several other hinist csv file which include the pixels information of corresponding images in different resolusion. In this project, we will focus on the information from the metadata. Also, when we creat the model, we will use the original images for higher resolusion, thus we will dismiss all the hmnist data this time. 

The data has seven different classes of skin cancer which are listed below :
1. Melanocytic nevi
2. Melanoma
3. Benign keratosis-like lesions
4. Basal cell carcinoma
5. Actinic keratoses
6. Vascular lesions
7. Dermatofibroma

In this project, I will try to train a model of 7 different skin cancer classes using Convolution Neural Network with Keras TensorFlow and then use it to predict the types of skin cancer with random images.
Here is the plan of the project step by step:



1. Import all the necessary libraries for this project
2. Make a dictionary of images and labels
3. Reading and processing the metadata
4. Process data cleaning
5. Exploring the data analysis
6. Train Test Split based on the data frame 
7. Creat and transfer the images to the corresponding folders 
8. Do image augmentation and generate extra images to the imbalanced skin types
9. Do data generator for training, validation, and test folders
10. Build the CNN model
11. Fitting the model
12. Model Evaluation
13. Visualize some random images with prediction


## 1. Import all the necessary libraries for this project
```
# import the necessary libraries for this project
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os, shutil
from glob import glob
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG19, inception_resnet_v2, xception
from tensorflow.keras import layers
from tensorflow.keras.layers import Dropout
from tensorflow.keras import models
from tensorflow.keras import optimizers
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report,confusion_matrix
import seaborn as sns

# import the libraries for dash plot
from jupyter_dash import JupyterDash
import dash
from dash import dcc
from dash import html 
import pandas as pd
import plotly.express as px
from dash.dependencies import Input, Output
app = JupyterDash(__name__)
server = app.server

```

## 2. Make a dictionary of images and labels
In this steps, I make the path for all the images and a dictionary for all types of skin cancers with full names.

```
path_dict = {os.path.splitext(os.path.basename(x))[0] :x for  x in glob(os.path.join('*', '*.jpg'))}
lesion_type_dict = {
    'nv': 'Melanocytic nevi',
    'mel': 'Melanoma',
    'bkl': 'Benign keratosis-like lesions ',
    'bcc': 'Basal cell carcinoma',
    'akiec': 'Actinic keratoses',
    'vasc': 'Vascular lesions',
    'df': 'Dermatofibroma'
}

```
## 3. Reading and processing the metadata
In this step, we have read the csv which had the information for all the patients and images. Afterthat, we made three more columns including the cancer type in full name, the label in skin cancers in digital and the path of image_id in the folder.

```
# read the metadata
meta = pd.read_csv('HAM10000_metadata.csv')

# generate new columns of type, lebel and path
meta['type'] = meta['dx'].map(lesion_type_dict.get)
meta['label'] = pd.Categorical(meta['type']).codes
meta['path'] = meta['image_id'].map(path_dict.get)
meta.head()
```

![fig1_meta](https://raw.githubusercontent.com/sachenl/project5/main/images/fig1_meta.png)

## 4. Process data cleaning
In this part, we check the missing values for each column and fill them. 

```
meta.isna().sum()
```
![fig2](https://raw.githubusercontent.com/sachenl/project5/main/images/fig2.png)

```
# fill the missing age with their mean.
meta['age'].fillna((meta['age'].mean()), inplace=True)
meta.isna().sum()
```
![](https://raw.githubusercontent.com/sachenl/project5/main/images/fig3.png)

## 5. Exploring the data analysis
In this part, we briefly explored different features of the dataset, their distributions and counts.

As there is some duplecate lesion_id which belong to same patient, all the features except the image_id for them are same with each other.  Thus, we first find and remove the duplex. 
```
# compare the unique values for lesion id and image id.
meta.lesion_id.nunique(), meta.image_id.nunique()
```
(7470, 10015)

```
# drop the duplication based on the lesion_id.
meta_patient = meta.drop_duplicates(subset=['lesion_id'])

meta_patient.head()
```
![](https://raw.githubusercontent.com/sachenl/project5/main/images/fig4.png)
```
# plot distribution of features 'dx', 'dx_type',  'sex', 'localization'.
feat = ['dx', 'dx_type',  'sex', 'localization']
plt.subplots(figsize=(11, 8))
for i, fea in enumerate(feat):
    length = len(feat)
    plt.subplot(2, 2, i+1)
    meta_patient[fea].value_counts().plot.bar(fontsize = 10)
    plt.ylabel('counts', fontsize = 15)
    plt.xticks()
    plt.title(fea, fontsize = 15)
    plt.tight_layout()
```

![](https://raw.githubusercontent.com/sachenl/project5/main/images/fig5.png)

We checked the distribution of columns 'dx', 'dx_type',  'sex', 'localization' for different patients. The graphs show that:
1. In dx features, the 'nv': 'Melanocytic nevi' case take more than 70% of the total cases. The number suggests that this dataset is an unbalanced dataset.
2. In dx_type features, the histogram suggests most of the cancer were confirmed in Follow-up and histo Histopathologic diagnoses.
3. The sex feature shows that the amount of male who had skin cancer is slight larger than female but still similar to each other.
4. The localization analysis shows that  lower extremity, back ,trunk abdomen and upper extremity are heavily compromised regions of skin cancer


```
# Creat dashboard to visualize the distribution of age for different types of skin cancer

app.layout = html.Div(children=[
    html.H1(children='Distribution of Age', style={'text-align': 'center'}),
    

    html.Div([
        html.Label(['Choose a graph:'],style={'font-weight': 'bold'}),
        dcc.Dropdown(
            id='dropdown',
            options=[
                {'label': 'all types', 'value': 'all'},
                {'label': 'nv', 'value': 'nv'},
                {'label': 'bkl', 'value': 'bkl'},
                {'label': 'mel', 'value': 'mel'},
                {'label': 'bcc', 'value': 'bcc'},
                {'label': 'akiec', 'value': 'akiec'},
                {'label': 'vasc', 'value': 'vasc'},
                {'label': 'df', 'value': 'df'}
                    ],
            value='all types',
            style={"width": "60%"}),
        
    html.Div(dcc.Graph(id='graph')),        
        ]),

])

@app.callback(
    Output('graph', 'figure'),
    [Input(component_id='dropdown', component_property='value')]
)
def select_graph(value):
    if value == 'all':
        fig = px.histogram(None , x= meta_patient['age'], nbins=20, labels={'x':value, 'y':'count'})
    
        return fig
    else:
        fig = px.histogram(None , x= meta_patient[meta_patient['dx'] == value]['age'], 
                           nbins=20,  labels={'x':value, 'y':'count'})
    
        return fig

if __name__ == '__main__':
    app.run_server(mode = 'inline')
```

![](https://raw.githubusercontent.com/sachenl/project5/main/images/ezgif.com-gif-maker.gif)


In general, most cancers happen between 35 to 70.  Age 45 is a high peak for patients to get a skin cancer.  Some types of skin cancer (vasc, nv) happen to those below 20, and others occur most after 30.


## 6. Train Test Split based on the data frame 
We split the dataset to training (70%), validation (10%) and testing (20%) by train_test_split.

```
df = meta.drop(columns='label')
target = meta['label']
X_train_a, X_test, y_train_a, y_test = train_test_split(df, target, test_size=0.2, random_state=123)
X_train, X_val, y_train, y_val = train_test_split(X_train_a, y_train_a, test_size=0.1, random_state=12)
X_train.shape, X_val.shape, X_test.shape
```
((7210, 9), (802, 9), (2003, 9))


## 7. Creat and transfer the images to the corresponding folders 
We created the subfolders containing the train, Val, and test folder. In addition, we created a folder for all types of skin cancers in each of the folders. Finally, We transferred the images to the corresponding folder based on the data frame and the path in each image ID.

```
# copy the images to correct folder according to the image_id in dataframe

new_dir = 'sub_folders2'
os.makedirs(new_dir)  # creat the subfolders
TVT = ['train', 'val', 'test']
lesion = lesion_type_dict.keys()
for first in TVT:
    temp_dir = os.path.join(new_dir, first)
    os.mkdir(temp_dir) # creat the train, val and test folders

    for sec in lesion:
        sec_dir = os.path.join(temp_dir, sec)
        os.mkdir(sec_dir) # creat the subfolders of all tpyes of cancers
        
        if first == 'train':
            source_df = X_train[X_train['dx'] == sec] # find the source of train dataset
        if first == 'val':
            source_df = X_val[X_val['dx'] == sec] # find the source of validation dataset
        elif first == 'test':
            source_df = X_test[X_test['dx'] == sec] # find the source of test dataset

        
        for source in source_df.path: # find the images to transfer
            shutil.copy(source, sec_dir)
        print("{} files copied to {}".format(len(source_df.path), sec_dir))
        
    
```

![](https://raw.githubusercontent.com/sachenl/project5/main/images/fig6.png)

## 8. Do image augmentation and generate extra images to the imbalanced skin types
The amounts of files in each training folder type tell us the images of nv are much higher than others. The imbalance of the training dataset might cause a high bias in model fitting. Thus we will generate some more images for other kinds of cancers. Here we use image augmentation to oversample the samples in all classes except nv. Here is a simple chart about the oversampling.

![](https://raw.githubusercontent.com/sachenl/project5/main/images/oversampling.png)

```
# We only need to fill more images to all class except nv
class_list = ['mel','bkl','bcc','akiec','vasc','df']

for cat in class_list:
    # creat temp folder for augmentaion
    temp_dir = 'temp'
    os.mkdir(temp_dir)
    img_dir = os.path.join(temp_dir, cat)
    os.mkdir(img_dir) 
    
    # copy the original images to temperate folder
    img_list = os.listdir('sub_folders2/train/'+cat)
     
    for image in img_list:
        source = os.path.join('sub_folders2/train/'+cat, image)
        dest = os.path.join(temp_dir)
        shutil.copy(source,img_dir)
    
    path = temp_dir
    save_path = 'sub_folders2/train/'+cat
    # set the parameters of image augmentation
    data_datagen = ImageDataGenerator(rescale=1./255,
                                       rotation_range=180, # randomly rotate images in the range (degrees, 0 to 180)
                                       width_shift_range=0.2, # randomly shift images horizontally (fraction of total width)
                                       height_shift_range=0.2, # randomly shift images vertically (fraction of total height)
                                       shear_range=0.2,# Randomly shear image 
                                       zoom_range=0.2, # Randomly zoom image 
                                       horizontal_flip=True, # randomly flip images
                                       vertical_flip=True,# randomly flip images
                                       fill_mode='nearest')

    
    batch_size = 50
    
    temp_generator = data_datagen.flow_from_directory(path,
                                                      save_to_dir = save_path,
                                                      save_format = 'jpg',
                                                      target_size = (224, 224),
                                                      batch_size=batch_size
                                                        )
    # Generate the temp images and add them to the training folders
    
    num_needed = 5000 
    num_cur =  len(os.listdir(img_dir))
    num_batches = int(np.ceil((num_needed - num_cur)/batch_size))
    
    for i in range (0, num_batches):
        imgs, label = next(temp_generator)
    # delete the temp folders after each transfer
    shutil.rmtree(temp_dir)
```

![](https://github.com/sachenl/project5/blob/main/images/fig7.png)

```
# check the files in each of the folders after image augmentation
class_list = ['nv', 'mel','bkl','bcc','akiec','vasc','df']

for cat in class_list:
    print(len(os.listdir('sub_folders/train/'+cat)))
```

![](https://github.com/sachenl/project5/blob/main/images/fig8.png)



















