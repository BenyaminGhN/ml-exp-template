import numpy as np
import SimpleITK as sitk
import cv2
import os
from sklearn.preprocessing import MinMaxScaler

class SingleImagePreprocessor:
    def __init__(self, target_size=(512, 512), add_channel=False,
                crop_ratio=(0.0, 0.0)):
        self.target_size = target_size
        self.add_channel = add_channel
        self.crop_ratio = crop_ratio

    def crop_image(self, image):

        h, w = image.shape
        zf_y, zf_x = self.crop_ratio

        # cropping images
        cropped_image = image[
            int(12*(zf_y*h)/10) : int(h-(zf_y*h)/2), 
            int((zf_x*w)/2) : int(w-(zf_x*w)/2)
        ]

        return cropped_image

    def prepare_image(self, image):

        img_arr = np.expand_dims(image, -1)
        img_arr_3_channel = img_arr.repeat(3, axis=-1)
        return img_arr_3_channel
    
    def rescale_image(self, img):
        img = img-img.min()
        img = img/img.max()
        return img

    def preprocess_image(self, image):

        # cropping to remove artifacts
        cropped_image = self.crop_image(image / image.max())

        # resizing image
        if cropped_image.shape[0:2] != self.target_size:
            w = self.target_size[1]
            h = self.target_size[0]
            image_array = cv2.resize(cropped_image, (w, h))
        else:
            image_array = cropped_image

        # enhancing images
        # image_array = self.enhance_img(image_array)

        # rescaling the image
        image_array = self.rescale_image(image_array)

        # convert gray-scale image to 3 channel ( for pretrained models )
        if self.add_channel:
            cleaned_img = self.prepare_image(image_array)
        else:
            cleaned_img = image_array

        return cleaned_img

    def enhance_img(self, img):
        # Getting the kernel to be used in Top-Hat
        filterSize =(3, 3)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, 
                                        filterSize)

        # Applying the Top-Hat operation
        tophat_img = cv2.morphologyEx(img, 
                                    cv2.MORPH_TOPHAT,
                                    kernel)
        
        # Applying the Black-Hat operation
        blackhat_img = cv2.morphologyEx(img, 
                                    cv2.MORPH_BLACKHAT,
                                    kernel)
        
        enhanced_img = img + tophat_img - blackhat_img
        return enhanced_img
    
    def read_dcm(self, dcm_file_path):

        volume = sitk.ReadImage(dcm_file_path, sitk.sitkInt32)
        volume = np.array(sitk.GetArrayFromImage(volume), dtype='float32')

        return volume[0]


class SingleImagePreprocessorSeg:
    def __init__(self, target_size=(512, 512), add_channel=False,
                 crop_ratio=(0.0, 0.0)):
        self.target_size = target_size
        self.add_channel = add_channel
        self.crop_ratio = crop_ratio

    def crop_image(self, image):

        h, w = image.shape
        zf_y, zf_x = self.crop_ratio

        # cropping images
        cropped_image = image[
            int((zf_y*h)/2) : int(h-(zf_y*h)/2), 
            int((zf_x*w)/2) : int(w-(zf_x*w)/2)
        ]
        return cropped_image

    def prepare_image(self, image):

        img_arr = np.expand_dims(image, -1)
        img_arr_3_channel = img_arr.repeat(3, axis=-1)
        return img_arr_3_channel
    
    def rescale_image(self, img):
        img = img-img.min()
        img = img/img.max()
        return img

    def preprocess_image(self, image):
        # cropping to remove artifacts
        cropped_image = self.crop_image(image / image.max())
        self.mask = self.crop_image(self.mask)

        # resizing image
        if cropped_image.shape[0:2] != self.target_size:
            w = self.target_size[1]
            h = self.target_size[0]
            image_array = cv2.resize(cropped_image, (w, h))
            self.mask = cv2.resize(self.mask, (w, h))
        else:
            image_array = cropped_image

        # enhancing images
        # image_array = self.enhance_img(image_array)

        # rescaling the image
        image_array = self.rescale_image(image_array)

        # convert gray-scale image to 3 channel ( for pretrained models )
        if self.add_channel:
            cleaned_img = self.prepare_image(image_array)
        else:
            cleaned_img = image_array

        self.mask = np.where(self.mask != 0, 1, 0).astype('float32')
        self.mask = np.expand_dims(self.mask, axis=-1)

        return cleaned_img, self.mask

    def enhance_img(self, img):
        # Getting the kernel to be used in Top-Hat
        filterSize =(3, 3)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, 
                                        filterSize)

        # Applying the Top-Hat operation
        tophat_img = cv2.morphologyEx(img, 
                                    cv2.MORPH_TOPHAT,
                                    kernel)
        
        # Applying the Black-Hat operation
        blackhat_img = cv2.morphologyEx(img, 
                                    cv2.MORPH_BLACKHAT,
                                    kernel)
        
        enhanced_img = img + tophat_img - blackhat_img
        return enhanced_img
    
    def read_img_mask(self, dcm_file_path, mask_file_path):

        self.read_mask(mask_file_path)

        volume = sitk.ReadImage(dcm_file_path, sitk.sitkInt32)
        volume = np.array(sitk.GetArrayFromImage(volume), dtype='float32')

        if self.mask.size == 0:
            self.mask = np.zeros(volume[0].shape)

        return volume[0]
    
    def read_mask(self, mask_file_path):

        if os.path.exists(mask_file_path):
            self.mask = cv2.imread(mask_file_path, cv2.IMREAD_GRAYSCALE)
        else:
            self.mask = np.zeros(0)