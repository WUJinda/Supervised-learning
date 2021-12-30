import cv2
import mahotas
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1 import ImageGrid
from matplotlib import pyplot as plt
from skimage import io


def visualise4(df):
    # df_4 = pd.DataFrame(df).head(4)
    df_4 = df.head(4)  # Recuperer les quatre premières lignes de la classe
    imgplt_list = []
    for index, row in df_4.iterrows():
        tmp = cv2.imread(row['path'])
        imgplt_list.append(np.array(tmp))
    if len(imgplt_list) < 4:
        print('not enough to 4')
        fig = plt.figure(figsize=(4., 2.))
        grid = ImageGrid(fig, 111,  # similar to subplot(111)
                         nrows_ncols=(1, 2),  # creates 1x2 grid of axes
                         axes_pad=0.1,  # pad between axes in inch.
                         )

        for ax, im in zip(grid, [imgplt_list[0], imgplt_list[1]]):
            # Iterating over the grid returns the Axes.
            ax.imshow(im)
        fig.suptitle(str(df_4['class'].iloc[1]))
        plt.show()
    else:

        fig = plt.figure(figsize=(4., 4.))
        fig.suptitle(str(df_4['class'].iloc[1]))
        grid = ImageGrid(fig, 111,  # similar to subplot(111)
                         nrows_ncols=(2, 2),  # creates 2x2 grid of axes
                         axes_pad=0.1,  # pad between axes in inch.
                         )

        for ax, im in zip(grid, [imgplt_list[0], imgplt_list[1], imgplt_list[2], imgplt_list[3]]):
            # Iterating over the grid returns the Axes.
            ax.imshow(im)

        plt.show()


# functions to extract features from the images
def fd_hu_moments(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature = cv2.HuMoments(cv2.moments(image)).flatten()
    return feature


# Define the function to extract Haralick features
def fd_Haralick(image):
    img = image[:, :, 0]  # Convert Image 2D to 3D
    features = mahotas.features.haralick(img)
    # features = mahotas.features.texture.haralick_features(img)
    feature = np.mean(features, axis=0)  # get mean value of cols
    return feature


# Define the function to extract image histogram
def color_histogram(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist_s = cv2.calcHist([hsv], [1], None, [256], [0, 256])
    hist_v = cv2.calcHist([hsv], [2], None, [256], [0, 256])
    df1 = pd.DataFrame(hist_s)
    df2 = pd.DataFrame(hist_v)
    hist = df1.append(df2)
    return hist


def feature_extraction(img_path):

    # read image from the path
    img = io.imread(img_path)
    # Extract three features corresponding to Hu moments, Haralick and Color histogram in HSV respectively.
    feature_hu = fd_hu_moments(img)
    feature_hrk = fd_Haralick(img)
    feature_hsv = color_histogram(img)
    df_hu = pd.DataFrame(feature_hu)
    df_hrk = pd.DataFrame(feature_hrk)
    df_hsv = feature_hsv
    # Merge into one dataframe of column 1
    features = df_hu.append(df_hrk.append(df_hsv, ignore_index=True), ignore_index=True)
    # print(features)
    # return features.values[:, 0]
    return features.values



if __name__ == '__main__':
    pass
