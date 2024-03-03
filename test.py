class BigResidual(Layer):
    def __init__(self, channels_in,kernel,**kwargs):
        super(BigResidual, self).__init__(**kwargs)
        self.channels_in = channels_in
        self.kernel = kernel

        self.reshape = Reshape((1,1, self.channels_in))
        self.conv1_3=Conv2D( self.channels_in,self.kernel,padding="same")
        self.leak1=LeakyReLU(alpha=LEAKY_MULTIPLYER)
        self.conv2_3=Conv2D( self.channels_in,(5, 5),padding="same")
        self.leak2=LeakyReLU(alpha=LEAKY_MULTIPLYER)
        self.conv3_3=Conv2D( self.channels_in,(7, 7),padding="same")
        self.leak3=LeakyReLU(alpha=LEAKY_MULTIPLYER)
        self.conv4_3=Conv2D( self.channels_in,self.kernel,padding="same")
        self.leak4=LeakyReLU(alpha=LEAKY_MULTIPLYER)
        self.add=Add()
        self.leak5=LeakyReLU(alpha=LEAKY_MULTIPLYER)
        self.drop=Dropout(0.45)
        self.bn=BatchNormalization()

    def call(self, x):
        first_conv = self.leak1(self.conv1_3(x))
        second_conv = self.leak2(self.conv2_3(x))
        third_conv = self.leak3(self.conv3_3(x))
        residual = self.bn(self.add([x, first_conv, second_conv, third_conv]))
        x = self.drop(self.leak5(residual))
        return x

    def compute_output_shape(self, input_shape):
        return input_shape