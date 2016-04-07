import numpy as np
import cv2
import os

try:
    IMREAD_UNCHANGED = cv2.CV_LOAD_IMAGE_UNCHANGED
except AttributeError:
    IMREAD_UNCHANGED = cv2.IMREAD_UNCHANGED


def init_cityscapes(cityscapes_root):
    '''
    Append the right paths to the sys paths in order to use the provided cityscapes scripts.
    - `cityscapes_root` the root folder of the dataset. Make sure to clone the cityscapescripts there too.
        https://github.com/mcordts/cityscapesScripts
    '''
    import sys
    cityscapes_scipts_dir = os.path.join(cityscapes_root, 'cityscapesScripts', 'scripts')
    sys.path.append(os.path.join(cityscapes_scipts_dir, 'helpers'))


def image_names(cityscapes_root, subset, citynames=False):
    '''
    Retrieve all image filenames for a specific subset from the cityscape dataset.
    - `cityscapes_root` the root folder of the dataset.
    - `subset` the subset to be loaded can be one of `train`, `test`, `val`, `train_extra`.
    '''
    image_folder = os.path.join(cityscapes_root, 'leftImg8bit', subset)
    cnames = []
    inames = []

    #Get all the images in the subfolders
    for city in os.listdir(image_folder):
        city_folder = os.path.join(image_folder, city)

        for fname in os.listdir(city_folder):
            if fname.endswith('.png'):
                inames.append(os.path.join(city_folder, fname))
                cnames.append(city)

    return (inames, cnames) if citynames else inames


def load_images(image_names, downscale_factor=1):
    '''
    Load all images for a set of image names as returned by `image_names`, optionally downscale the images.
    - `image_names` the list of image names to load.
    - `downscale_factor` the factor with which the images will be downscaled.

    Returns the images in an uint8 array of shape (N,3,H,W).
    '''
    H, W = 1024//downscale_factor, 2048//downscale_factor
    X = np.empty((len(image_names), 3, H, W), np.uint8)

    #Get all the images in the subfolders
    for i, imname in enumerate(image_names):
        im = cv2.imread(imname)
        if downscale_factor != 1:
            im = cv2.resize(im, (W, H))
        X[i] = np.rollaxis(im[:,:,::-1], 2)  # cv2 to theano (BGR to RGB and HWC to CHW)

    return X


def downscale_labels(factor, labels, threshold, dtype=np.int8):
    '''
    Downscale a label image. Each `factor`x`factor` window will be mapped to a single pixel.
    If the the majority label does not have a percentage over the `threshold` the pixel will
    be mapped to -1.
    - `factor` the factor with which the images will be downscaled.
    - `labels` the input labels.
    - `threshold` the required part of the majority be a valid label [0.0, 1.0].
    - `dtype` the datatype of the returned label array. The default allows labels up to 128.
    '''
    m = np.min(labels)
    M = np.max(labels)
    if m < -1:
        raise Exception('Labels should not have values below -1')
    h,w = labels.shape
    h_n = int(np.ceil(float(h)/factor))
    w_n = int(np.ceil(float(w)/factor))
    label_sums = np.zeros((h_n, w_n, M+2))
    for y in xrange(0, h):
        for x in xrange(0, w):
            label_sums[y/factor, x/factor, labels[y,x]] +=1

    hit_counts = np.sum(label_sums,2)

    label_sums = label_sums[:,:,:-1]
    new_labels = np.argsort(label_sums, 2)[:,:,-1].astype(dtype)
    counts = label_sums.reshape(h_n*w_n, M+1)
    counts = counts[np.arange(h_n*w_n),new_labels.flat]
    counts = counts.reshape((h_n, w_n))

    hit_counts *=threshold
    new_labels[counts < hit_counts] = -1
    return new_labels

def load_labels(image_names, downscale_factor = None, label_downscale_threshold = 0.0, fine=True):
    '''
    Load all label images for a set of rgb image names.
    `image_names` the rgb image names for which the ground truth labels should be loaded.
    `downscale_factor` the factor with which the labels are downscaled. None results in the orignal size.
    `label_downscale_threshold` the majority label ratio needed in order to achieve a valid label.
    `fine` wether to load the fine labels (True), or the coarse labels (False). Fine labels are only available for a subset.
    '''
    #Needed for the label definitions from CS.
    import labels as cs_labels

    #Create a map to map between loaded labels and training labels.
    label_map = np.asarray([t.trainId if t.trainId != 255 else -1 for t in cs_labels.labels], dtype=np.int8)

    labels = []
    #Find the corresponding label images
    for l in image_names:
        l = l.replace('leftImg8bit', 'gtFine' if fine else 'gtCoarse',1)
        l = l.replace('leftImg8bit', 'gtFine_labelIds' if fine else 'gtCoarse_labelIds')
        l_im = cv2.imread(l, IMREAD_UNCHANGED)
        if l_im is None:
            raise ValueError("Couldn't load image {}".format(l))
        l_im_mapped = label_map[l_im]
        if downscale_factor is not None:
            l_im_mapped = downscale_labels(downscale_factor, l_im_mapped, label_downscale_threshold)
        labels.append(l_im_mapped)

    return labels