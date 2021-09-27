from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB6

train_dir = ".../train"
val_dir = ".../val"
test_dir = ".../test"

train_datagen = ImageDataGenerator(   # normalization already done by EfficientNet
    rotation_range = 180,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True,
    vertical_flip = True
)

valid_datagen = ImageDataGenerator() # no data augmentation for validation images

input_size = 384 
batch_size = 16

train_generator = train_datagen.flow_from_directory(
    directory = train_dir,
    target_size = (input_size, input_size),
    batch_size = batch_size,
    class_mode = 'categorical'
)

val_generator = valid_datagen.flow_from_directory(
    directory = val_dir,
    target_size = (input_size, input_size),
    batch_size = batch_size,
    class_mode = 'categorical'
)

base = EfficientNetB6(weights = 'imagenet', include_top = False, input_shape = (input_size, input_size, 3))

base.trainable = False   # keeping the weights of the pre-trained network

inputs = keras.Input(shape = (input_size, input_size, 3))
x = base(inputs, training = False)
x = layers.GlobalAveragePooling2D()(x)
outputs = layers.Dense(5, activation = 'softmax')(x)
model = keras.Model(inputs, outputs)

adam = keras.optimizers.Adam(lr = 0.001)  
model.compile(loss = 'categorical_crossentropy', optimizer = adam, metrics = [keras.metrics.AUC(curve = "PR")])

history = model.fit(
    train_generator,
    steps_per_epoch = len(train_generator.classes) // batch_size,
    epochs = 30,
    validation_data = val_generator,
    validation_steps = len(val_generator.classes) // batch_size
)

test_datagen = ImageDataGenerator()

test_generator = test_datagen.flow_from_directory(
    directory = test_dir,
    target_size = (input_size, input_size),
    batch_size = batch_size,
    class_mode = 'categorical'
)

model.evaluate(test_generator)

model.save('EfficientNetB6.h5')
