import pandas as pd
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, GlobalAveragePooling2D, Activation
from tensorflow.keras.optimizers import SGD

data_df = pd.read_csv('Subsample.csv')
path = "F:\XUEXI\FISE3\INFO\Option\AlgoANDA\projet\Subsample_Histo"
print(data_df.shape)
print(data_df.head())
train_df, val_df = train_test_split(data_df, test_size=0.2, random_state=1)
print(val_df.head())
datagen = ImageDataGenerator(rescale=1. / 255)

img_width = 700
img_height = 460
bs = 8

train_generator = datagen.flow_from_dataframe(dataframe=train_df, directory=path,
                                               target_size=(img_width, img_height), shuffle=True)

validation_generator = datagen.flow_from_dataframe(dataframe=val_df, directory=path,
                                                    target_size=(img_width, img_height), shuffle=False)

# def build_masked_loss():#the saturated pixels (label=-1) are removed from the loss evaluation
#     def custom_rmse(y_true, y_pred,mask=-1):
#         mask = K.cast(K.not_equal(y_true, mask), K.floatx())
#         return K.sqrt(K.mean(K.square(mask*(y_pred - y_true)), axis=-1)+0.000001) # adding this epsilon to avoid Nan when the gray output is saturated
#     return custom_rmse
#
# model = Sequential()
# model.add(Conv2D(32, (3, 3), strides=(1, 1), input_shape=(224, 224, 3), padding='valid', activation='relu',
#                  kernel_initializer='uniform'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))
# model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))
# model.add(Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))
# model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Flatten())
# model.add(Dense(256, activation='relu'))
# model.add(Dropout(0.25))
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.25))
# model.add(Dense(38, activation="sigmoid"))
# opt = SGD(learning_rate=0.01)
# model.compile(loss='categorical_crossentropy', optimizer=opt)
# EPOCHS = 50
#
# H = model.fit(train_generator, validation_data=validation_generator, epochs=EPOCHS)
# plt.figure()
# plt.plot(H.history["loss"], label="train_loss")
# plt.plot(H.history["val_loss"], label="val_loss")
# plt.title("Training Loss ")
# plt.xlabel("Epoch #")
# plt.ylabel("Loss")
# plt.legend()
# x = model.predict(validation_generator)
# print(x)
