import cv2
import numpy as np
from keras import layers, models
from keras.preprocessing.image import img_to_array, load_img
from keras.utils import Sequence
from scipy import ndimage


class DataGenerator(Sequence):
    """
    A data generator class that extends keras.utils.Sequence.

    Ref: https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
    """

    def __init__(
        self,
        images,
        image_dir,
        labels,
        label_dir,
        augmentation=None,
        preprocessing=None,
        batch_size=8,
        dim=(256, 256, 3),
        shuffle=True,
    ):
        """
        Initialization method for the DataGenerator class.

        Args:
            images (list): List of image filenames.
            image_dir (str): Directory where images are stored.
            labels (list): List of label filenames.
            label_dir (str): Directory where labels are stored.
            augmentation (callable, optional): Function for augmenting the images. Defaults to None.
            preprocessing (callable, optional): Function for preprocessing the images. Defaults to None.
            batch_size (int, optional): Number of samples per gradient update. Defaults to 8.
            dim (tuple, optional): Dimensions of the images. Defaults to (256, 256, 3).
            shuffle (bool, optional): Whether to shuffle the images between epochs. Defaults to True.
        """
        self.dim = dim
        self.images = images
        self.image_dir = image_dir
        self.labels = labels
        self.label_dir = label_dir
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        """
        Denotes the number of batches per epoch.

        Returns:
            int: Number of batches per epoch.
        """
        return int(np.floor(len(self.images) / self.batch_size))

    def __getitem__(self, index):
        """
        Generate one batch of data.

        Args:
            index (int): Index of the batch to generate.

        Returns:
            tuple: A batch of images and labels.
        """
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [k for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        """
        Updates indexes after each epoch.
        If shuffle is True, shuffles the indexes.
        """
        self.indexes = np.arange(len(self.images))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        """
        Generates data containing batch_size samples.

        Args:
            list_IDs_temp (list): List of indexes for the images to include in the batch.

        Returns:
            tuple: A batch of images and labels.
        """
        batch_imgs = list()
        batch_labels = list()

        for i in list_IDs_temp:
            # Store sample
            img = load_img(self.image_dir + "/" + self.images[i], target_size=self.dim)
            img = img_to_array(img) / 255.0

            # Store class
            label = load_img(self.label_dir + "/" + self.labels[i], target_size=self.dim)
            label = img_to_array(label)[:, :, 0]
            label = label != 0
            label = ndimage.binary_erosion(ndimage.binary_erosion(label))
            label = ndimage.binary_dilation(ndimage.binary_dilation(label))
            label = np.expand_dims((label) * 1, axis=2)

            # apply augmentations
            if self.augmentation:
                sample = self.augmentation(image=img, mask=label)
                img, label = sample["image"], sample["mask"]

            # apply preprocessing
            if self.preprocessing:
                sample = self.preprocessing(image=img, mask=label)
                img, label = sample["image"], sample["mask"]

            batch_imgs.append(img)
            batch_labels.append(label)

        return np.array(batch_imgs, dtype=np.float32), np.array(batch_labels, dtype=np.float32)


class Segment:
    """
    The Segment class represents a U-Net model for semantic segmentation.

    Attributes:
        filters (int): The number of filters for the first convolutional block. This number is doubled after each max pooling layer in the encoder part and halved after each deconvolutional block in the decoder part.
        model (Model): The U-Net model.
    """

    def __init__(self, filters=64):
        """
        Initializes a new instance of the Segment class.

        Parameters:
            filters (int): The number of filters for the first convolutional block. Defaults to 64.
        """
        self.filters = filters
        self.model = self._create_unet(self.filters)

    def _create_unet(self, filters):
        """
        Creates a U-Net model.

        The U-Net model consists of an encoder (downsampler) and decoder (upsampler).
        In the encoder, the image is repeatedly convolved and downsampled until a bottleneck layer.
        In the decoder, the image is repeatedly convolved and upsampled, and the output from corresponding layers in the encoder are added to the upsampled image.

        Parameters:
            filters (int): The number of filters for the first convolutional block.

        Returns:
            Model: The U-Net model.
        """

        # downsampling
        input_layer = layers.Input(shape=(256, 256, 3), name="image_input")
        conv1 = self._conv_block(input_layer, nfilters=filters)
        conv1_out = layers.MaxPooling2D(pool_size=(2, 2))(conv1)
        conv2 = self._conv_block(conv1_out, nfilters=filters * 2)
        conv2_out = layers.MaxPooling2D(pool_size=(2, 2))(conv2)
        conv3 = self._conv_block(conv2_out, nfilters=filters * 4)
        conv3_out = layers.MaxPooling2D(pool_size=(2, 2))(conv3)
        conv4 = self._conv_block(conv3_out, nfilters=filters * 8)
        conv4_out = layers.MaxPooling2D(pool_size=(2, 2))(conv4)
        conv4_out = layers.Dropout(0.5)(conv4_out)
        conv5 = self._conv_block(conv4_out, nfilters=filters * 16)
        conv5 = layers.Dropout(0.5)(conv5)

        # upsampling
        deconv6 = self._deconv_block(conv5, residual=conv4, nfilters=filters * 8)
        deconv6 = layers.Dropout(0.5)(deconv6)
        deconv7 = self._deconv_block(deconv6, residual=conv3, nfilters=filters * 4)
        deconv7 = layers.Dropout(0.5)(deconv7)
        deconv8 = self._deconv_block(deconv7, residual=conv2, nfilters=filters * 2)
        deconv9 = self._deconv_block(deconv8, residual=conv1, nfilters=filters)
        output_layer = layers.Conv2D(filters=1, kernel_size=(1, 1), activation="sigmoid")(deconv9)

        # using sigmoid activation for binary classification
        model = models.Model(inputs=input_layer, outputs=output_layer, name="Unet")

        return model

    def _conv_block(self, tensor, nfilters, size=3, padding="same", initializer="he_normal"):
        """
        Defines a convolutional block with two Conv2D layers, each followed by BatchNormalization and ReLU activation.

        Parameters:
        tensor (tf.Tensor): The input tensor.
        nfilters (int): The number of filters for the Conv2D layers.
        size (int, optional): The kernel size for the Conv2D layers. Defaults to 3.
        padding (str, optional): The padding method for the Conv2D layers. Defaults to "same".
        initializer (str, optional): The initializer for the Conv2D layers. Defaults to "he_normal".

        Returns:
        tf.Tensor: The output tensor after applying the convolutional block.
        """
        x = layers.Conv2D(
            filters=nfilters,
            kernel_size=(size, size),
            padding=padding,
            kernel_initializer=initializer,
        )(tensor)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Conv2D(
            filters=nfilters,
            kernel_size=(size, size),
            padding=padding,
            kernel_initializer=initializer,
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)

        return x

    def _deconv_block(self, tensor, residual, nfilters, size=3, padding="same", strides=(2, 2)):
        """
        Defines a deconvolutional block with a Conv2DTranspose layer followed by a concatenation with the residual, and a convolutional block.

        Parameters:
        tensor (tf.Tensor): The input tensor.
        residual (tf.Tensor): The residual tensor to concatenate with the output of the Conv2DTranspose layer.
        nfilters (int): The number of filters for the Conv2DTranspose layer.
        size (int, optional): The kernel size for the Conv2DTranspose layer. Defaults to 3.
        padding (str, optional): The padding method for the Conv2DTranspose layer. Defaults to "same".
        strides (tuple, optional): The strides for the Conv2DTranspose layer. Defaults to (2, 2).

        Returns:
        tf.Tensor: The output tensor after applying the deconvolutional block.
        """
        y = layers.Conv2DTranspose(
            nfilters, kernel_size=(size, size), strides=strides, padding=padding
        )(tensor)
        y = layers.concatenate([y, residual], axis=3)
        y = self._conv_block(y, nfilters)

        return y

    def info(self):
        """
        Prints the summary of the U-Net model. The summary includes the layers in the model,
        the shape of the output of each layer, the number of parameters in each layer, and
        the total number of parameters in the model.
        """
        self.model.summary()

    def load(self, path):
        """
        Loads the weights of the U-Net model from a file.

        Parameters:
            path (str): The path to the file containing the weights.
        """
        self.model.load_weights(path)

    def predict(self, img, postprocess=True):
        """
        Predicts the mask for an image using the U-Net model.

        The image is resized to 256x256 pixels, normalized to the range [0, 1], and expanded to
        include a batch dimension before it's passed to the model. The output of the model is
        thresholded at 0.5 to create a binary mask, and then resized back to the original size
        of the image.

        If postprocess is True, the mask is postprocessed to remove noise, small objects, and
        to smooth the staircase edges that might be introduced during or after the resizing.

        Parameters:
            img (np.array): The input image.
            postprocess (bool, optional): Whether to postprocess the mask. Defaults to True.

        Returns:
            np.array: The predicted mask.
        """
        input = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA) / 255.0
        input = np.expand_dims(input, axis=0)
        mask = (self.model.predict(input) > 0.5).astype(np.uint8).reshape(256, 256)
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)

        if postprocess:
            mask = self._postprocess(mask)

        return mask

    def _postprocess(self, mask):
        """
        Postprocesses a mask to remove noise and small objects, and to smooth the staircase
        edges that might be introduced during or after the resizing.

        The mask is dilated and blurred to merge nearby objects, and then eroded and blurred
        to remove small objects and smooth the edges of the objects.

        Parameters:
            mask (np.array): The input mask.

        Returns:
            np.array: The postprocessed mask.
        """
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
        out = cv2.dilate(mask, kernel, iterations=3)
        out = cv2.medianBlur(out, 35)
        out = cv2.erode(out, kernel, iterations=6)
        out = cv2.medianBlur(out, 35)
        out = cv2.dilate(out, kernel, iterations=3)
        out = cv2.medianBlur(out, 35)

        return out

    def compile(self, optimizer, loss, metrics):
        """
        Compiles the U-Net model with an optimizer, a loss function, and a list of metrics.

        Parameters:
            optimizer (tf.keras.optimizers.Optimizer): The optimizer to use.
            loss (tf.keras.losses.Loss): The loss function to use.
            metrics (list): A list of metrics to use.
        """
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def train(self, train_data, val_data, callbacks, epochs=100, model_checkpoint="best.h5"):
        """
        Trains the U-Net model on the training data.

        Parameters:
            train_data (tf.data.Dataset): The training data.
            val_data (tf.data.Dataset): The validation data.
            epochs (int): The number of epochs to train the model.
            batch_size (int): The batch size to use during training.
        """
        return self.model.fit(
            train_data,
            steps_per_epoch=len(train_data),
            epochs=epochs,
            callbacks=callbacks,
            validation_data=val_data,
            validation_steps=len(val_data),
        )

    def evaluate(self, test_data):
        """
        Evaluates the U-Net model on the test data.

        Parameters:
            test_data (tf.data.Dataset): The test data.

        Returns:
            dict: A dictionary containing the evaluation metrics.
        """
        return self.model.evaluate(test_data, steps=len(test_data))
