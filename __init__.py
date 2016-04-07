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


def load_images_from_folder(cityscapes_root, subset, downscale_factor = 1):
    '''
    Load a specific subset from the cityscape dataset, optionally downscale the images.
    - `cityscapes_root` the root folder of the dataset.
    - `subset` the subset to be loaded can be one of train, test, val, train_extra.
    - `downscale_factor` the factor with which the images will be downscaled.

    Returns the a list of cities, the image filenames and the images.
    '''
    image_folder = os.path.join(cityscapes_root,'leftImg8bit', subset)
    city_names = []
    image_names = []
    images = []

    #Get all the subfolders
    city_folders = [c for c in os.listdir(image_folder) if os.path.isdir(os.path.join(image_folder,c))]

    #Get all the images in the subfolders
    for c in city_folders:
        city_folder = os.path.join(image_folder, c)
        for i in [i for i in os.listdir(city_folder) if os.path.isfile(os.path.join(city_folder,i))]:
            city_names.append(c)
            image_names.append(os.path.join(city_folder,i))
            im = cv2.imread(image_names[-1])[:,:,::-1]
            images.append(cv2.resize(im,(im.shape[1]//downscale_factor, im.shape[0]//downscale_factor)))

    return city_names, image_names, images


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