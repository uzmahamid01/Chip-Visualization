#!/usr/bin/env python
# coding: utf-8

# In[1]:


#starting the program


# In[2]:


# Embedding Functionality
class PathObject:
  def __init__(self, output_path: str, data_path: str, train_embed_path: str, test_embed_path: str, train_img_path: str, test_img_path: str):
    self.output_path: str = output_path
    self.data_path: str = data_path
    self.train_embed_path: str = train_embed_path
    self.test_embed_path: str = test_embed_path
    self.train_image_path: str = train_img_path
    self.test_image_path: str = test_img_path

from enum import Enum

class EnumExtended(Enum):
    @classmethod
    def list(cls):
        return list(map(lambda c: c.value, cls))

class EvalCriteria(EnumExtended):
    EUCLIDEAN = 1
    COSINE = 2
    
class ChipName(EnumExtended):
    AMD = 'amd'
    APPLE = 'ammple'
    ATHEROS = 'atheros'
    ATT = 'att'
    BROAD = 'broadcom'
    DALLAS = 'dallas'
    FIJITSU = 'fijistu'
    HITACHI = 'itachi'
    INFINEON = 'infineon'
    INTEL = 'intel'
    MEDIATEK = 'mediatek'
    MITSUBISHI = 'mitsubishi'
    MOTOROLLA = 'motorolla'
    NEC = 'nec'
    NUVTON = 'nuvoton'
    OKI = 'oki'
    PANASONIC = 'panasonic'
    PHILIPS = 'philips'
    QUALCOM = 'qualcomm'
    SANYO = 'sanyo'
    SHARP = 'sharp'
    SIEMENS = 'siemens'
    SONY = 'sony'
    TEXAS = 'texas'
    TOSHIBA = 'toshiba'
    VIA = 'via'
    YAMAHA = 'yamaha'

def get_chip_names():
    return ChipName.list()

#  uzma, hamid --> {uzma: hamid}
def read_delimted_data_as_map(text_with_delimter_list):
    text_delimted_map = dict()
    for text_with_delimter in text_with_delimter_list:
        parts = text_with_delimter.strip('\n').split(',')
        text_delimted_map[parts[0]] =  int(parts[1])
    return text_delimted_map
    

def read_file_as_list(path_to_file: str):
    file_context_list: list[str] = None
    with open(path_to_file) as f:
        file_context_list = f.readlines()
    return file_context_list
    


# In[3]:


import scipy.io

def load_mat_file(file_path: str)-> dict():
    mat = scipy.io.loadmat(file_path)
    return mat

# Not working as it is giving codec error [Need to know which encoding to use]
def load_mat_file_numpy(file_path: str):
    import numpy as np
    mat = np.loadtxt(file_path)
    print(type(mat))
    return mat


# In[4]:


from scipy.spatial.distance import cdist
import numpy
def get_pairwise_distance(X: numpy.ndarray, Y: numpy.ndarray, distance_mode:str = 'euclidean'):
    return cdist(X, Y, distance_mode)


# In[5]:


# [4, 0, 2, 1] -> [3, 0, 2, 1]
# [0, 1, 2, 4]
def get_sorted_array_indices(array: numpy.ndarray): 
    sorted_indices = numpy.argsort(array, kind='mergesort', axis=1) # Sorts along axis 1 [X axis]
    sorted_array = numpy.sort(array, axis = 1)
    return (sorted_array, sorted_indices)


# In[6]:


from PIL import Image               # to load images
from IPython.display import display # to display images
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

def display_image_from_file(filepath: str):
    pil_im = Image.open(filepath)
    display(pil_im)

def display_image_with_title(filepath: str, title: str):
    try:
        # depicting the visualization
        pil_im = Image.open(filepath)
        # displaying the title
        plt.title(title,
                  fontsize='10',
                  backgroundcolor='green',
                  color='white')
        #plt.imshow(pil_im)
        display(pil_im)
        display.clear_output(wait=True)
        time.sleep(1)
    except KeyboardInterrupt:
        print("KB Interrupt")


# In[7]:


import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time
get_ipython().run_line_magic('matplotlib', 'inline')

def show_images_side_by_side(images_path_map: dict):
    plt.figure(figsize=(20,10))
    columns = 4
    images = list(images_path_map.keys())
    num_rows = int(len(images)/(columns)) + 1
    for i, image_path in enumerate(images):
        plt.subplot(num_rows, columns, i+1)
        pil_im = Image.open(image_path)
        plt.imshow(pil_im)


# In[8]:


def find_idx_in_array(values, searchval):
    idx = numpy.where(values == searchval)[0]
    return idx


# In[9]:


# Get Recall Functionality
import numpy
from scipy import stats
def get_recall(paths: PathObject , 
               chip_name_list: list[str] = get_chip_names(), 
               evaluation_criteria: EvalCriteria = EvalCriteria.EUCLIDEAN, 
               number_images_similarity: int = 7, 
               visualize: bool = True, 
               debug: bool = False):
    # Get Paths from the Paths object
    train_embed_path: str = paths.train_embed_path
    test_embed_path: str = paths.test_embed_path
    train_img_path: str = paths.train_image_path
    test_img_path: str = paths.test_image_path
    
    # read from same file used during train/test, which stores tuples of (image_name, manufacturer_label)
    
    # Prepare the image name from training list and testing list
    train_image_names: list[str] =  list(read_delimted_data_as_map(read_file_as_list(paths.data_path + "/train_image_list.txt")).keys())
    test_image_names: list[str] = list(read_delimted_data_as_map(read_file_as_list(paths.data_path + "/test_image_list.txt")).keys())
        
    
    # Loading the mat files for Train and Test Embedding
    test_embed: dict = load_mat_file(test_embed_path)
    train_embed: dict = load_mat_file(train_embed_path)

    train_feat: numpy.ndarray  = train_embed['feat']
    train_label: numpy.ndarray = train_embed['label'][0]
    test_feat: numpy.ndarray = test_embed['feat']
    test_label: numpy.ndarray = test_embed['label'][0]
        
        
    if debug:
        print(f"Test Image Name: {test_label}")
        
    train_sample_size: int = train_label.size
    test_sample_size: int = test_label.size
    
    # Measure the disance based on the distance used [metric]
    #Initialize the matrices
    distance_matrix = numpy.zeros((test_sample_size, train_sample_size))
    label_matrix = numpy.zeros((test_sample_size, train_sample_size))
    index_matrix = numpy.zeros((test_sample_size, train_sample_size))
    
    if debug:
        print(f"The Test Sample Size is: {test_sample_size}")
        print(f"The Train Sample Size is: {train_sample_size}")
        print(f"Train labels are of size: {train_label.shape}")
    
    # Calculate the distances of 
    for i in range(0, test_sample_size):
        print(f"Computing the distance from test -> train of {i} out of {test_sample_size}")
        # We need the row and then reshape so that we get the distance from al the training features
        test_feat_ith = test_feat[i,:].reshape(-1, 1).T
        if EvalCriteria.EUCLIDEAN == evaluation_criteria:
            unsorted_dist = get_pairwise_distance(test_feat_ith, train_feat)
        elif EvalCriteria.COSINE == evaluation_criteria:
            unsorted_dist = get_pairwise_distancea(test_feat_ith, train_feat, 'cosine')
        else:
            unsorted_dist = get_pairwise_distance(test_feat_ith, train_feat)
        
        # Sort the distances to find the labels of the samples we got
        sorted_distance, sorted_indices = get_sorted_array_indices(unsorted_dist)
        
        if debug: 
            print(f"Unsorted distance shape: {unsorted_dist.shape}")
            print(f"Sorted distance shape: {sorted_distance.shape}")
            print(f"Sorted distance: {sorted_distance}")
            print(f"Indices are of shape: {sorted_indices.shape}")
            print(f"Sorted indices: {sorted_indices}")
            print(f"Sum of sorted indices: {sum(sorted_indices[0,:])}")

        distance_matrix[i,:] = sorted_distance
        if debug: 
            print(f"After putting the value we have distance matrix as: {distance_matrix[0,:]}")
        label_matrix[i,:] = train_label[sorted_indices]
        index_matrix[i,:] = sorted_indices
    
    # Now using the visualization tool and windows, show the similiar images
    all_test_recall = []
    nn_list = []
    
    test_recall = numpy.zeros((test_sample_size, 1))
    
    # Recall Loop
    start_loop = 0
    end_loop = test_sample_size
    if visualize:
        start_loop = 400
        end_loop = 401
    for i in range (start_loop, end_loop):    
        test_image_name = test_image_names[i]    
        test_image_label = test_label[i] 
        
        print(f"Checking the matched labels for image: {test_image_name} Chipset: {chip_name_list[int(test_image_label)-1]}. Image {i} of {test_sample_size}")
        
        print(f"Test Image Label is: {test_image_label}")
            
        matched_indices = index_matrix[i, :]
        matched_labels = label_matrix[i, :]
        
        # Extract only limited number of matches from all the matches [top 7 matches]
        
        matched_indices = matched_indices[0:number_images_similarity]
        matched_labels = matched_labels[0:number_images_similarity]
        
        print(f"Matched Indices: {matched_indices}")
        print(f"Matched Lables: {matched_labels}")
        
        # Store the test image and matching 7 images to show in one plot
        image_path_name_map = {}
        
        if visualize:
            test_img_path: str = os.path.join(paths.data_path , 'test' , test_image_name)
            # Since the Matlab code and enumeration starts from 1, subtracting 1 to make it 0 indexed
            # display_image_with_title(test_img_path, chip_name_list[int(test_image_label)-1])
            image_path_name_map[test_img_path] = chip_name_list[int(test_image_label)-1]
        
        # Get the evaluation matrices
        if EvalCriteria.EUCLIDEAN == evaluation_criteria:
            idx = find_idx_in_array(matched_labels, test_image_label)
            if idx.any:
                test_recall[i] = 1
        
        elif EvalCriteria.COSINE == evaluation_criteria:
            # Get the mode using the scipy
            frequent_label, count = stats.mode(matched_labels)
            most_freq_label = frequent_label[0]
            
            idx = find_idx_in_array(most_freq_label, test_image_label)
            if idx.any:
                test_recall[i] = 1
        else:
            print("Doesn't support any other evaluation criteria")
            exit(1)
        
        # Show top K images in a plot
        for j in range(0, number_images_similarity):
                matched_image_name = train_image_names[int(matched_indices[j])]
                matched_image_label = matched_labels[j]
                matched_image_name_path: str = os.path.join(paths.data_path, 'train', matched_image_name)
                image_path_name_map[matched_image_name_path] = chip_name_list[int(matched_image_label)-1]
                # display_image_with_title(matched_image_name_path, chip_name_list[int(matched_image_label)-1])
        
        if visualize:
            show_images_side_by_side(image_path_name_map)
        
    all_test_recall = sum(test_recall)/test_sample_size
    print(f"Total recall of the system is: {all_test_recall}")
    nn_list = [] # Don't know what this is
    return all_test_recall, nn_list
        
        
    


# In[10]:


# Evaluate Embedding function 
def eval_embedding(paths: PathObject, chips: list[str] = ChipName.list(), 
                   eval_criteria: EvalCriteria = EvalCriteria.EUCLIDEAN):
    return get_recall(paths=paths, chip_name_list=chips, evaluation_criteria=eval_criteria)


# In[14]:


# Main Function
import os

os.path.join

model_name = "resnet50"
epoch_name = '225'
learning_rate = 'lr.000001'

#base_path: str = "C://Users/abham/workspace/ChipSimilarity" # Path where all the code is stored
base_path: str = "/Users/master-node/Desktop/ChipSimilarity" # Path where all the code is stored
    
data_path: str = os.path.join(base_path, "data_v4")
output_path : str = os.path.join(base_path, "output_" +  model_name + '_datav4_' + learning_rate)
train_image_path: str = os.path.join(data_path, 'train')
test_image_path: str = os.path.join(data_path, 'test')
train_embed_path: str = os.path.join(output_path, 'train', model_name + '_feat_' + epoch_name + '_train.mat')
test_embed_path: str = os.path.join(output_path, 'test', model_name + '_feat_' + epoch_name + '_test.mat')

paths: PathObject = PathObject(output_path=output_path, data_path=data_path, train_embed_path=train_embed_path, 
                               test_embed_path=test_embed_path, train_img_path=train_image_path, test_img_path=test_image_path)

## Generate path for saving the recongnition results


## Mkdir if it doesn't exist

eval_criteria = EvalCriteria.EUCLIDEAN

## Call eval Embeddings

eval_embedding(paths=paths, eval_criteria=eval_criteria)


# In[12]:


# TESTING CODE AFTER THISSSSS #######


# In[13]:


from PIL import Image               # to load images
from IPython.display import display # to display images

pil_im = Image.open(data_path + '/test/000229.jpg')
display(pil_im)


# In[ ]:


test_file_path = paths.test_embed_path
train_file_path = paths.train_embed_path

X = load_mat_file(test_file_path)
Y = load_mat_file(train_file_path)

test_feat  = X['feat']
test_label = X['label']
train_feat = Y['feat']
train_label = Y['label']

# print(test_feat.shape)
# print(test_feat[2,:].shape)
# print(train_feat.shape)

a = test_feat[1, :]
print(a.shape)
print (a)
test_feat_new = a.reshape((-1,1)).T

print(test_feat_new.shape)

print(test_feat_new)

# print(test_feat_new)

from scipy.spatial.distance import cdist
distance = cdist(test_feat_new, train_feat)
print(get_pairwise_distance(test_feat_new, train_feat))

print(distance.shape)

print(distance)


# In[ ]:


# import numpy as np
# from scipy import stats
# sort along the first axis
a = np.array([1,2, 3, 3, 3, 4, 4, 4,5])
print(np.where(a == 3)[0])
search_val = 3
print(find_idx_in_array(a, search_val))
if find_idx_in_array(a, search_val).any:
    print("Got it")
# print(a)
# arr1 = np.sort(a, axis = 0)
# arr2 = np.argsort(a, kind = 'mergesort', axis = 0)
# print ("Along first axis : \n", arr1)
# print( arr2)
# distance_matrix = np.zeros((2,2))
# print (distance_matrix)

# distance_matrix[1,:] = arr1[0]
# print(distance_matrix)


# In[ ]:


import numpy as np
from scipy.spatial.distance import cdist

a = np.array([[0.6787, 0.7431], [0.7577, 0.3922]])
print(a)
b = np.array([[0.655, 0.0318], [0.1712, 0.2769], [0.7060, 0.0462]])
print(b)
c = a[0, :].reshape(-1, 1).T
print(c.shape)
print(c)

d = cdist(c,b,'euclidean')

print(d)
print(d.shape)

print(get_pairwise_distance(c, b))

# results =  cdist(xx,yy,'euclidean')
# print (results)
# print(results.shape)


# In[ ]:


array = get_pairwise_distance(c, b)


# In[ ]:


sorted_indices = numpy.argsort(array, kind='mergesort', axis=1) # Sorts along axis 0
sorted_array = numpy.sort(array, axis = 1)


# In[ ]:


print(sorted_indices)
print(sorted_array)


# In[ ]:


import numpy
import sys

print("Python version")
print (sys.version)
print("Version info.")
print (sys.version_info)

