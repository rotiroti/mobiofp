import cv2
import numpy as np
import rembg
from keras import layers, models
from ultralytics import YOLO


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
        output_layer = layers.Conv2D(
            filters=1, kernel_size=(1, 1), activation="sigmoid"
        )(deconv9)

        # using sigmoid activation for binary classification
        model = models.Model(inputs=input_layer, outputs=output_layer, name="Unet")

        return model

    def _conv_block(
        self, tensor, nfilters, size=3, padding="same", initializer="he_normal"
    ):
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

    def _deconv_block(
        self, tensor, residual, nfilters, size=3, padding="same", strides=(2, 2)
    ):
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
        mask = cv2.resize(
            mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR
        )

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

    # def jaccard_distance_loss(y_true, y_pred, smooth=100):


#     """
#     Calculates the Jaccard distance loss between the true and predicted labels.

#     Parameters:
#     y_true (tf.Tensor): The true labels.
#     y_pred (tf.Tensor): The predicted labels.
#     smooth (int, optional): A smoothing factor to prevent division by zero. Defaults to 100.

#     Returns:
#     tf.Tensor: The Jaccard distance loss.
#     """
#     intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
#     sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
#     jac = (intersection + smooth) / (sum_ - intersection + smooth)

#     return (1 - jac) * smooth


# def dice_coef(y_true, y_pred):
#     """
#     Calculates the Dice coefficient between the true and predicted labels.

#     Parameters:
#     y_true (tf.Tensor): The true labels.
#     y_pred (tf.Tensor): The predicted labels.
#     smooth (int, optional): A smoothing factor to prevent division by zero. Defaults to 1.

#     Returns:
#     tf.Tensor: The Dice coefficient.
#     """
#     y_true_f = K.flatten(y_true)
#     y_pred_f = K.flatten(y_pred)
#     intersection = K.sum(y_true_f * y_pred_f)

#     return (2.0 * intersection + K.epsilon()) / (
#         K.sum(y_true_f) + K.sum(y_pred_f) + K.epsilon()
#     )


class Detect:
    """
    A class used to detect objects in an image using the YOLO model.

    This class provides methods to load the YOLO model from a checkpoint, predict the objects in an image,
    get the bounding box of the detected objects, extract the region of interest (ROI) from the image,
    create a mask of the ROI, and visualize the detection results.

    Attributes:
        _model (YOLO): The YOLO model used for object detection.
        _checkpoint (str): The path to the checkpoint file for the YOLO model.

    Methods:
        info(): Print the information of the YOLO model.
        predict(image: np.ndarray, safe: bool = False): Predict the objects in an image.
        bbox(result: YOLO): Get the bounding box of the detected objects.
        roi(result: YOLO, image: np.ndarray): Extract the region of interest (ROI) from the image.
        mask(result: YOLO, image: np.ndarray): Create a mask of the ROI.
        show(result: YOLO): Visualize the detection results.
    """

    _model = None
    _checkpoint = None

    def __init__(self, checkpoint: str) -> None:
        """
        Initialize the Detect class.

        This method loads the YOLO model from a checkpoint.

        Args:
            checkpoint (str): The path to the checkpoint file for the YOLO model.
        """
        self._model = YOLO(checkpoint)
        self._checkpoint = checkpoint

    def info(self):
        """
        Print the information of the YOLO model.

        This method uses the `info` method of the YOLO model to print its information.
        """
        self._model.info()

    def predict(self, image: np.ndarray, safe: bool = False):
        """
        Predict the objects in an image.

        This method uses the YOLO model to predict the objects in an image. If the `safe` argument is True,
        it creates a new instance of the YOLO model from the checkpoint for each prediction.

        Args:
            image (np.ndarray): The input image.
            safe (bool, optional): Whether to create a new instance of the YOLO model for each prediction. Defaults to False.

        Returns:
            YOLO: The prediction result.

        Raises:
            ValueError: If no objects are detected in the image.
        """
        local_model = self._model

        if safe:
            local_model = YOLO(self._checkpoint)

        results = local_model.predict(image, conf=0.85)

        if len(results) == 0:
            raise ValueError("No objects detected in the image.")

        result = results[0]

        return result

    def bbox(self, result: YOLO) -> [int, int, int, int]:
        """
        Get the bounding box of the detected objects.

        This method extracts the bounding box of the detected objects from the prediction result.

        Args:
            result (YOLO): The prediction result.

        Returns:
            list: A list of four integers representing the bounding box (x1, y1, x2, y2).
        """
        boxes = result.boxes.xyxy.tolist()
        x1, y1, x2, y2 = boxes[0]

        return int(x1), int(y1), int(x2), int(y2)

    def roi(self, result: YOLO, image: np.ndarray) -> np.ndarray:
        """
        Extract the region of interest (ROI) from the image.

        This method extracts the ROI from the image based on the bounding box of the detected objects.

        Args:
            result (YOLO): The prediction result.
            image (np.ndarray): The input image.

        Returns:
            np.ndarray: The ROI.
        """
        x1, y1, x2, y2 = self.bbox(result)

        return image[int(y1) : int(y2), int(x1) : int(x2)]

    def mask(self, result: YOLO, image: np.ndarray) -> np.ndarray:
        """
        Create a mask of the ROI.

        This method creates a mask of the ROI by removing the background and applying some post-processing steps.

        Args:
            result (YOLO): The prediction result.
            image (np.ndarray): The input image.

        Returns:
            np.ndarray: The mask of the ROI.
        """
        roi = self.roi(result, image)

        # Use Rembg to remove background
        mask = rembg.remove(roi, only_mask=True)

        # Post-process mask
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        mask = cv2.GaussianBlur(
            mask, (5, 5), sigmaX=2, sigmaY=2, borderType=cv2.BORDER_DEFAULT
        )
        mask = np.where(mask < 127, 0, 255).astype(np.uint8)

        return mask

    def show(self, result: YOLO) -> np.ndarray:
        """
        Visualize the detection results.

        This method uses the `plot` method of the prediction result to visualize the detection results.

        Args:
            result (YOLO): The prediction result.

        Returns:
            np.ndarray: The visualization of the detection results.
        """
        return result.plot()
