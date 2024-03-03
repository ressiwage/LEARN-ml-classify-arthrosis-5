


def model(path):
    from keras.optimizers import Adam
    from keras.layers import Conv2D, MaxPooling2D, Add, GlobalAveragePooling2D, Reshape, DepthwiseConv2D, BatchNormalization, LeakyReLU, Dropout, Activation, Dropout, Flatten, Dense, Layer
    from keras.models import Sequential
    
    LEAKY_MULTIPLYER = 0.005
    class BigResidual(Layer):
        def __init__(self, channels_in,kernel,**kwargs):
            super(BigResidual, self).__init__(**kwargs)
            self.channels_in = channels_in
            self.kernel = kernel
            self.depconv = DepthwiseConv2D(self.channels_in,self.kernel,padding="same")
            self.gap = GlobalAveragePooling2D()
            self.reshape = Reshape((1,1, self.channels_in))
            self.layer1=Conv2D( self.channels_in,self.kernel,padding="same")
            self.leak1=LeakyReLU(alpha=LEAKY_MULTIPLYER)
            self.layer2=Conv2D( self.channels_in,self.kernel,padding="same")
            self.leak2=LeakyReLU(alpha=LEAKY_MULTIPLYER)
            self.layer3=Conv2D( self.channels_in,self.kernel,padding="same")
            self.leak3=LeakyReLU(alpha=LEAKY_MULTIPLYER)
            self.layer4=Conv2D( self.channels_in,self.kernel,padding="same")
            self.leak4=LeakyReLU(alpha=LEAKY_MULTIPLYER)
            self.layer5=Add()
            self.layer6=LeakyReLU(alpha=LEAKY_MULTIPLYER)
            self.drop=Dropout(0.4)
            self.bn=BatchNormalization()
        def call(self, x):
            first_layer = self.layer1(x)
            first_conv = self.leak2(self.layer2(first_layer))
            second_conv = self.leak3(self.layer3(first_conv))
            x = self.leak1(self.layer4(second_conv))
            residual = self.bn(self.layer5([x, first_layer, first_conv, second_conv]))
            x = self.drop(self.layer6(residual))
            return x
        def compute_output_shape(self, input_shape):
            return input_shape
        def get_config(self):
            config = super(BigResidual, self).get_config()
            config.update({
                'channels_in': self.channels_in,
                'kernel': self.kernel
            })
            return config


    model = Sequential()
    model.add(Conv2D(400, (1, 1), input_shape=(112, 112, 3)))
    #model.add(MaxPooling2D(2,2))
    #model.add(LeakyReLU(alpha=LEAKY_MULTIPLYER))

    #model.add(Conv2D(400, (1, 1)))
    #model.add(LeakyReLU(alpha=LEAKY_MULTIPLYER))
    model.add(BigResidual(400, (3,3)))
    model.add(MaxPooling2D(3,3))

    model.add(Conv2D(64, (1, 1)))
    model.add(LeakyReLU(alpha=LEAKY_MULTIPLYER))
    model.add(BigResidual(64, (3,3)))

    model.add(Conv2D(32, (1, 1)))
    model.add(LeakyReLU(alpha=LEAKY_MULTIPLYER))
    model.add(BigResidual(32, (3,3)))

    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
    model.add(Dense(64))
    model.add(LeakyReLU(alpha=LEAKY_MULTIPLYER))

    model.add(Dropout(0.1))
    model.add(Dense(40))
    model.add(LeakyReLU(alpha=LEAKY_MULTIPLYER))

    model.add(Dropout(0.1))
    model.add(Dense(30))
    model.add(LeakyReLU(alpha=LEAKY_MULTIPLYER))

    model.add(Dropout(0.1))
    model.add(Dense(20))
    model.add(LeakyReLU(alpha=LEAKY_MULTIPLYER))
    model.add(Dropout(0.5))
    model.add(Dense(3, activation='softmax'))


    model.compile(loss='sparse_categorical_crossentropy',
                optimizer=Adam(1e-05),
                metrics=['accuracy'])


    model.load_weights(path)
    model.build()
    return model