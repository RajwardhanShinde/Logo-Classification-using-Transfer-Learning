from keras.applications import MobileNet
from keras.layers import Dense, GlobalAveragePooling2D
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.callbacks import ModelCheckpoint, EarlyStopping
import gc
gc.collect()
#from google_images_download import google_images_download

#Use google_images_download to download images 
#I have downloaded images for 4 classes you can download more
 
#response = google_images_download.googleimagesdownload()
#arguments = {"keywords":"apple", "limit":50, "print_urls":False, "format":"jpg", "size":">400*300"}
#paths = response.download(arguments)
#arguments = {"keywords":"samsung", "limit":50, "print_urls":False, "format":"jpg", "size":">400*300"}
#paths = response.download(arguments)
#arguments = {"keywords":"google", "limit":50, "print_urls":False, "format":"jpg", "size":">400*300"}
#paths = response.download(arguments)
#arguments = {"keywords":"microsoft", "limit":50, "print_urls":False, "format":"jpg", "size":">400*300"}
#paths = response.download(arguments)
#-----------------

epochs = 5
train_batch_size = 14
#val_batch_size = 6
img_rows, img_cols = 224, 224


base_model = MobileNet(weights="imagenet", include_top=False)

#You can freeze all base_models layers if u want
#for layer in base_model.layers:
    #layer.trainable=False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation="relu")(x)
x = Dense(1024, activation="relu")(x)
x = Dense(512, activation="relu")(x)
preds = Dense(4, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=preds)

#Freeze first 20 layers and train other
for layer in model.layers[:20]:
    layer.trainable=False
for layer in model.layers[20:]:
    layer.trainable=True

  
train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

#You can also use validation to test 
#val_datagen = ImageDataGenerator(#preprocessing_function=preprocess_input)

train_gen = train_datagen.flow_from_directory('Logos/',
                                             target_size=(img_rows,img_cols),
                                             color_mode="rgb",
                                             batch_size=train_batch_size,
                                             class_mode="categorical",
                                             shuffle=True)

#val_gen = val_datagen.flow_from_directory('/home/raj/Transfer_Learning/Val',
                                          #target_size=(img_rows,img_cols),
                                          #batch_size=val_batch_size,
                                          #class_mode="categorical",
                                          #shuffle=True)

#Define checkpoint and earlystop to monitor loss and save model with less loss
checkpoint = ModelCheckpoint("Weights.h5",
                             monitor="loss",
                             mode="min",
                             save_best_only=True,
                             verbose=1)

earlystop = EarlyStopping(monitor="loss",
                          min_delta=0,
                          patience=5,
                          verbose=1,
                          restore_best_weights=True)

callbacks = [earlystop, checkpoint]

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

step_size_train = train_gen.n//train_gen.batch_size
model.fit_generator(generator=train_gen,
                    callbacks=callbacks,
                   # validation_data=val_gen,
                   steps_per_epoch=step_size_train,
                   #validation_steps=15,
                   epochs=epochs)

#save model
model.save("Weights.h5")
