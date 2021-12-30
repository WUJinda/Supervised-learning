import numpy as np
import pandas as pd
import cv2
import os
import mahotas
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# from projet.cnnTraining import cnn_model
from projet.method import *
from skimage import io
from sklearn.preprocessing import MinMaxScaler

'''
Image feature extraction
Before trying any machine learning model, 
you need to extract features from your image, 
meaning you need to convert each image into a set of vectors (features) best describing the image. 
These vectors are then used as the input for your ML model. 
We propose a small example of feature extraction from images using the open-CV library on Python. 
You are expected to explore this part more deeply for the project. 
We also guide you for the beginning.

Some imports
Here are some imports that will be useful for the practical.

    cv2 : pip install opencv-python
'''

# Data visualisation
# get images list from folder
path = 'F:/XUEXI/FISE3/INFO/Option/AlgoANDA/projet/Subsample_Histo'
images_list = os.listdir(path)
img_path = []
print(f'These are {len(images_list)} images: ')
for name in images_list:
    img_path.append(path + '/' + name)
    # print(name)

# images visualisation
img_01 = io.imread(img_path[0])
# io.imshow(img_01)
# io.show()
'''
Dataset building
Now we can build a pandas DataFrame to store the information about the images. 
All the informations are contained in the filename of each image.

For example, SOB_B_TA-14-4659-40-001.png is the image 1, at magnification factor 40X, of a benign tumor of 
type tubular adenoma, original from the slide 14-4659, which was collected by procedure SOB.

Using the function str.split create new columns in the dataframe 
corresponding to the class, subclass and slide from each image. 
Print informations about the number of images of each class and each subclass.
'''
# get info from the filenames and enrich the dataframe
# by adding columns 'class', 'subclass' and 'slide'
class_list = []
subclass_list = []
slide_list = []
for name in images_list:
    # print('class', name.split('_', 1)[0])
    class_list.append(name.split('_', 1)[0])
    # print('subclass', name.split('_', 1)[1].split('-', 1)[0])
    subclass_list.append(name.split('_', 1)[1].split('-', 1)[0])
    # print('slide', name.split('-', 1)[1].split('-40-')[0])
    slide_list.append(name.split('-', 1)[1].split('-40-')[0])

df = pd.DataFrame({'class': class_list,
                   'subclass': subclass_list,
                   'slide': slide_list,
                   'path': img_path})
print(df)

# plot 4 images for each class

groups = df.groupby(df['subclass'])
df_BF = groups.get_group('B_F')
df_BTA = groups.get_group('B_TA')
df_MDC = groups.get_group('M_DC')
df_MLC = groups.get_group('M_LC')
df_MMC = groups.get_group('M_MC')
df_MPC = groups.get_group('M_PC')

# visualise4(df_BF)
# visualise4(df_BTA)
# visualise4(df_MDC)
# visualise4(df_MLC)
# visualise4(df_MMC)
# visualise4(df_MPC)


print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
'''
Features extraction

The selected set of features should be a small set whose values efficiently discriminate among 
patterns of the different classes, but are similar for patterns within the same class.
Here we will calculate 3 global features on each image. 
To do so, we can use functions from OpenCV and mahotas libraries:
    Hu moments: https://docs.opencv.org/2.4/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html
    Haralick: https://mahotas.readthedocs.io/en/latest/features.html
    Color histogram in HSV: https://docs.opencv.org/master/dd/d0d/tutorial_py_2d_histogram.html
'''

# parameters for the features extraction
bins = 8
fixed_size = tuple((175, 115))
features = []
test_f = []
for path in img_path:
    features.append(feature_extraction(path))
    test_f.append(feature_extraction(path)[:, 0])

fe = pd.DataFrame(test_f)
print(fe)
result = pd.concat([df, fe], axis=1, join='outer')
print(result)
# feature_test_hu = fd_hu_moments(img_01)
# print(feature_test_hu)
# print('----------------------------------------------------------------')
# # print(pd.DataFrame(feature_test_hu))
# feature_test_hrk = fd_Haralick(img_01)
# print(feature_test_hrk)
# print('----------------------------------------------------------------')
# feature_test_hsv = color_histogram(img_01)
# print(pd.DataFrame(feature_test_hsv).values[:, 0])
# feature_test_np_hsv = pd.DataFrame(feature_test_hsv).values[:, 0]
# print('----------------------------------------------------------------')
# print(type(feature_test_hu))
# print(type(feature_test_hrk))
# print(type(feature_test_np_hsv))
# features = np.append(feature_test_hu, feature_test_hrk, feature_test_np_hsv)
# df_hu = pd.DataFrame(feature_test_hu)
# df_hrk = pd.DataFrame(feature_test_hrk)
# df_hsv = feature_test_hsv
# features = df_hu.append(df_hrk.append(df_hsv, ignore_index=True), ignore_index=True)
# print(features)
print('----------------------------------------------------------------')
'''
Add here the code to calculate and normalize the features for each image in the list to extend 
the DataFrame defined earlier. Note that there are 7 values for the Hu moments, 
13 for the Haralick features and 512 for the histogram. At the end, the DataFrame contains 532 values 
between 0 and 1 for the global features calculated.
'''

# features extraction
features_extraction = []
scaler = MinMaxScaler()
for feature in features:
    data_max_min = scaler.fit_transform(feature)
    features_extraction.append(data_max_min[:, 0])
# print(features_extraction)
df_features = pd.DataFrame(features_extraction)
df_final = pd.concat([df, df_features], axis=1, join='outer')  # 横向合并
print(df_final)
# print(fe)
# df.insert(4, 'features', features_extraction)
# print(df)
# 把上面得到的特征，写到list中，7+13+512（532）个特征存到list里算所有的特征

'''
2. Model training
    2.1 - Principal component analysis (PCA)
            A good way to understand and evaluate the difficulty of separating the two classes is to visually look at 
            the data. Principal component analysis (PCA) consists in a linear dimensionality reduction using 
            Singular Value Decomposition of the data to project it to a lower dimensional space.

            By projecting the dataset in a 2-dimensions space, they can be visualised to identify their distribution 
            according to their class.

            Use the PCA function of sklearn to view the dataset in 2D: sklearn.decomposition.PCA

'''

# from sklearn.decomposition import PCA

# classify with test/train split regardless of the slide
X = df_final.iloc[:, 4:]  # remove the 4 first columns of the data
y = df_final.iloc[:, 2]  # keep only the second column

# perform PCA to see class distribution
pca = PCA(n_components=2)
newData = pca.fit_transform(X)
print(newData)
x_pca = np.dot(newData, pca.components_)
plt.title("PCA dimensionality reduction")
plt.xlim(-0.000015, 0.000015)
plt.ylim(-0.000015, 0.000015)
plt.scatter(x_pca[:, 0], x_pca[:, 1])
plt.show()
'''
An popular alternative to PCA is TSNE (T-distributed Stochastic Neighbor Embedding)
PCA is a mathematical approach and tries to separate points as far as possible based on highest variance
TSNE is a probabilistic approach and tries to group points as close as possible based on probability 
that two close points came from the same population distribution
More information on differences between PCA and TSNE can be found HERE(https://towardsdatascience.com/pca-vs-tsne-el-cl%C3%A1sico-9948181a5f87
Use the TNSE function of sklearn to view the dataset in 2D: sklearn.manifold.TNSE
'''

# from sklearn.manifold import TSNE

# calculate and show TSNE on the data
tsne = TSNE(n_components=2, init='pca', random_state=0)
X_tsne = tsne.fit_transform(X)
print("One data dimension is {}. Other data dimension is {}".format(X.shape[-1], X_tsne.shape[-1]))
print(X_tsne)

plt.title("TSNE dimensionality reduction")
plt.scatter(X_tsne[:, 0], X_tsne[:, 1])
plt.show()
'''
Ready for Machine Learning ? From here, it is up to you!

A small checklist of questions to ask yourself before you start !

    What kind of task do I have to solve (supervised ? regression or classification ?)
    Hence, what are the models at my disposal ?
    What are the suitable metrics that I should use to evaluate my model ?
    
    
Bonus !
    Explore and test a deep learning model (CNN are probably the most appropriate ones). 
    Python offers interesting framework to ease the use of these models: TensorFlow (Keras) 
    or Pytorch are the most common ones.

    You can perform multi-class classification as you have different types of malignant 
    or begnin tumor (MC, PC, etc.). This information can be found in the name of the file 
    (after the letter B or M. Use this as a label instead of the B vs. M first task.
'''

df_final.to_csv('Subsample.csv')
# test_result = cnn_model(df, path)
