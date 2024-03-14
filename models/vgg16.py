from keras.models import Sequential
from keras.optimizers import SGD
import keras.layers as layers
import keras.applications as applications


def get_vgg16(output_classes):
    vgg16 = Sequential(layers=[
        applications.VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3)),
        layers.GlobalAveragePooling2D(),
        layers.Dense(output_classes, activation='softmax')
    ])

    vgg16.summary()

    vgg16.compile(optimizer=SGD(learning_rate=0.001, momentum=0.9), loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return vgg16
