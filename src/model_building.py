import tensorflow as tf
import numpy as np
from omegaconf.dictconfig import DictConfig
tfk = tf.keras
tfkl = tfk.layers
K = tfk.backend

from .utils import (f1, iou_coef, dice_coef, dice_loss)

class ResNet50():
    def __init__(self, config: DictConfig):
        self.config = config

    def get_compiled_model(self, class_weights = None) -> tfk.Model:
        mb_conf = self.config.model_builder
        input_shape = mb_conf.input_shape
        weights = mb_conf.weights
        activation = mb_conf.activation
        n_classes = mb_conf.n_classes
        input = tfkl.Input(shape=input_shape)
        base_model = tfk.applications.resnet.ResNet50(
            include_top=False,
            input_tensor=input,
            input_shape=input_shape,
            weights=weights,
            pooling="avg",
        )
        x = base_model.output
        # x = Dropout(0.15)(x)

        prediction = tfkl.Dense(n_classes, activation=activation, name='opg_surgery')(x)
        model = tfk.Model(inputs=input, outputs=prediction)

        # compiling
        optimizer = tfk.optimizers.legacy.Adam(learning_rate=2.5e-6)
        metrics = [
            tfk.metrics.SensitivityAtSpecificity(0.8),
            tfk.metrics.AUC(curve="PR", name="AUC of Precision-Recall Curve"),
            tfk.metrics.FalseNegatives(),
            tfk.metrics.FalsePositives(),
            tfk.metrics.TrueNegatives(),
            tfk.metrics.TruePositives(),
            f1
        ]

        if class_weights is not None:
            loss = self.__get_weighted_loss(weights=class_weights)
        else:
            loss = K.binary_crossentropy
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        return model
    
    @staticmethod
    def __get_weighted_loss(weights):
        def weighted_loss(y_true, y_pred):
            loss_output = K.mean((weights[:,0]**(1-y_true))*(weights[:,1]**(y_true))*K.binary_crossentropy(y_true, y_pred), axis=-1)
            return loss_output
        return weighted_loss

class BaselineModel():
    def __init__(self, config: DictConfig):
        self.config = config

    def get_compiled_model(self, class_weights = None) -> tfk.Model:
        mb_conf = self.config.model_builder
        input_shape = mb_conf.input_shape
        activation = mb_conf.activation
        n_classes = mb_conf.n_classes
        model = self._get_model_arch(input_shape=input_shape, 
                                    activation=activation,
                                    n_classes=n_classes)

        # compiling
        optimizer = tfk.optimizers.legacy.Adam(learning_rate=2.5e-6)
        metrics = [
            tfk.metrics.SensitivityAtSpecificity(0.8),
            tfk.metrics.AUC(curve="PR", name="AUC of Precision-Recall Curve"),
            tfk.metrics.FalseNegatives(),
            tfk.metrics.FalsePositives(),
            tfk.metrics.TrueNegatives(),
            tfk.metrics.TruePositives(),
            f1
        ]

        if class_weights is not None:
            loss = self.__get_weighted_loss(weights=class_weights)
        else:
            loss = K.binary_crossentropy

        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        return model

    def _get_model_arch(self, input_shape, activation, n_classes):
        input = tfkl.Input(shape=input_shape)
        x = tfkl.Conv2D(32, (3, 3), activation='relu')(input)
        x = tfkl.MaxPooling2D((2, 2))(x)
        x = tfkl.Conv2D(64, (3, 3), activation='relu')(x)
        x = tfkl.MaxPooling2D((2, 2))(x)
        x = tfkl.Conv2D(64, (3, 3), activation='relu')(x)
        x = tfkl.MaxPooling2D((2, 2))(x)
        x = tfkl.Conv2D(128, (3, 3), activation='relu')(x)
        x = tfkl.MaxPooling2D((2, 2))(x)
        # x = tfkl.Conv2D(128, (3, 3), activation='relu')(x)
        # x = tfkl.MaxPooling2D((2, 2))(x)
        x = tfkl.Flatten()(x)
        x = tfkl.Dense(512, activation='relu')(x)
        x = tfkl.Dense(128, activation='relu')(x)
        x = tfkl.Dense(64, activation='relu')(x)
        prediction = tfkl.Dense(n_classes, activation=activation)(x)

        model = tfk.Model(inputs=input, outputs=prediction)
        return model

    @staticmethod
    def __get_weighted_loss(weights):
        def weighted_loss(y_true, y_pred):
            loss_output = K.mean((weights[:,0]**(1-y_true))*(weights[:,1]**(y_true))*K.binary_crossentropy(y_true, y_pred), axis=-1)
            return loss_output
        return weighted_loss
    
class UNet():
    def __init__(self, config: DictConfig):
        self.config = config

    def get_compiled_model(self, class_weights = None) -> tfk.Model:

        mb_conf = self.config.model_builder
        input_shape = mb_conf.input_shape
        weights = mb_conf.weights
        activation = mb_conf.activation
        model = self._get_model_arch(input_shape=input_shape, 
                                    activation=activation)

        # compiling
        optimizer = tfk.optimizers.legacy.Adam(learning_rate=2.5e-6)
        metrics = [
            # tfk.metrics.MeanIoU(num_classes=1), 
            # m_iou(2).mean_iou, 
            iou_coef, 
            dice_coef, 
            f1, 
            'acc'
        ]

        loss = dice_loss
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        return model
    
    def _get_model_arch(self, input_shape, activation):
        def down_block(x, filters, use_maxpool = True):
            x = tfkl.Conv2D(filters, 3, padding= 'same')(x)
            x = tfkl.BatchNormalization()(x)
            x = tfkl.LeakyReLU()(x)
            x = tfkl.Conv2D(filters, 3, padding= 'same')(x)
            x = tfkl.BatchNormalization()(x)
            x = tfkl.LeakyReLU()(x)
            if use_maxpool == True:
                return  tfkl.MaxPooling2D(strides= (2,2))(x), x
            else:
                return x
        def up_block(x,y, filters):
            x = tfkl.UpSampling2D()(x)
            x = tfkl.Concatenate(axis = 3)([x,y])
            x = tfkl.Conv2D(filters, 3, padding= 'same')(x)
            x = tfkl.BatchNormalization()(x)
            x = tfkl.LeakyReLU()(x)
            x = tfkl.Conv2D(filters, 3, padding= 'same')(x)
            x = tfkl.BatchNormalization()(x)
            x = tfkl.LeakyReLU()(x)
            return x
            
        def unet(input_size = (256, 256, 3), *, classes, dropout):
            filter = [16, 32, 64, 128, 256]
            # encode
            input = tfkl.Input(shape = input_size)
            x, temp1 = down_block(input, filter[0])
            x, temp2 = down_block(x, filter[1])
            x, temp3 = down_block(x, filter[2])
            x, temp4 = down_block(x, filter[3])
            x = down_block(x, filter[4], use_maxpool= False)
            # decode 
            x = up_block(x, temp4, filter[3])
            x = up_block(x, temp3, filter[2])
            x = up_block(x, temp2, filter[1])
            x = up_block(x, temp1, filter[0])
            x = tfkl.Dropout(dropout)(x)
            output = tfkl.Conv2D(classes, 1, activation=activation)(x)
            model = tfk.Model(input, output, name = 'unet')
            return model
        
        return unet(input_size = input_shape, classes=1, dropout=0.0)
 