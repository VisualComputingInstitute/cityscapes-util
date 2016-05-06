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
        if im is None:
            raise ValueError("Couldn't load image {}".format(imname))
        if downscale_factor != 1:
            im = cv2.resize(im, (W, H), interpolation=cv2.INTER_AREA)
        X[i] = np.rollaxis(im[:,:,::-1], 2)  # cv2 to theano (BGR to RGB and HWC to CHW)

    return X


def downscale_labels(labels, f, threshold, dtype=np.int8):
    '''
    Downscale a label image. Each `factor`x`factor` window will be mapped to a single pixel.
    If the the majority label does not have a percentage over the `threshold` the pixel will
    be mapped to -1.
    - `labels` the input labels.
    - `f` the factor with which the images will be downscaled. Can be an integer or a (y,x) tuple.
    - `threshold` the required part of the majority be a valid label [0.0, 1.0].
    - `dtype` the datatype of the returned label array. The default allows labels up to 128.
    '''
    fy, fx = f if isinstance(f, (list, tuple)) else (f,f)
    H,W = labels.shape
    assert (H % fy) == 0 and (W % fx) == 0, "image size must be divisible by factor!"
    h,w = H//fy, W//fx

    m = np.min(labels)
    M = np.max(labels)
    assert -1 <= m, 'Labels should not have values below -1'

    # Oh come on now `troisdorf_000000_000073_gtCoarse_labelIds` you little cunt.
    if m == M:
        return np.full((h,w), m, dtype)

    # Count the number of occurences of the labels in each "fy x fx cell"
    label_sums = np.zeros((h, w, M+2))
    mx, my = np.meshgrid(np.arange(w), np.arange(h))
    for dy in range(fy):
        for dx in range(fx):
            label_sums[my, mx, labels[dy::fy,dx::fx]] += 1
    label_sums = label_sums[:,:,:-1]  # "Don't know" don't count

    # Use the highest-occurence label
    new_labels = np.argsort(label_sums, 2)[:,:,-1].astype(dtype)

    # But turn "uncertain" cells into "don't know" label.
    counts = label_sums[my, mx, new_labels]
    hit_counts = np.sum(label_sums, 2) * threshold
    new_labels[counts <= hit_counts] = -1

    return new_labels


def upsample(im, factor):
    """ Very fast upsampling of two last axes of `im`age by integer `factor`. """
    return np.repeat(np.repeat(im, factor, axis=-1), factor, axis=-2)


def load_labels(image_names, fine=True, preprocess=None):
    '''
    Load all label images for a set of rgb image names.
    - `image_names` the rgb image names for which the ground truth labels should be loaded.
    - `fine` wether to load the fine labels (True), or the coarse labels (False). Fine labels are only available for a subset.
    - `preprocess` is a function which, given a single label image returns a new label image.
        For example, you can use `lambda x: downscale_labels(x, 8, 0.5)`.
    '''
    #Needed for the label definitions from CS.
    import labels as cs_labels

    #Create a map to map between loaded labels and training labels.
    label_map = np.asarray([t.trainId if t.trainId != 255 else -1 for t in cs_labels.labels], dtype=np.int8)

    y = None

    #Find the corresponding label images
    for i, name in enumerate(image_names):
        name = name.replace('leftImg8bit', 'gtFine' if fine else 'gtCoarse',1)
        name = name.replace('leftImg8bit', 'gtFine_labelIds' if fine else 'gtCoarse_labelIds')
        im = cv2.imread(name, IMREAD_UNCHANGED)
        if im is None:
            raise ValueError("Couldn't load image {}".format(name))
        im_mapped = label_map[im]
        if preprocess is not None:
            im_mapped = preprocess(im_mapped)

        # The first image determines the size of the output array for all.
        # This allows preprocessing to fully determine size and dtype.
        if y is None:
            H, W = im_mapped.shape
            y = np.empty((len(image_names), H, W), im_mapped.dtype)

        y[i] = im_mapped

    return y


def label_names():
    from labels import labels

    mapping = {l.trainId: l.name for l in labels}
    mapping[255] = mapping[-1] = 'ignore'

    return [mapping[i] for i in range(19)] + [mapping[-1]]


def translate(imgs, mapping, remaining=-1, output=None):
    """
    Given an array `imgs` and an iterable of pairs `(a,b)` in `mapping`,
    this translates all `a`s present in `imgs` into `b`s and all remaining
    entries into `remaining`s. Optionally write the result into the existing
    `output` array.
    """
    if output is None:
        output = np.full_like(imgs, remaining)
    else:
        output[:] = remaining

    for src, dst in mapping:
        output[imgs==src] = dst

    return output


def id2trainId(im, unk=-1):
    # Yeah, manually bitch.
    # It's not like they will change these numbers anytime soon, that'd be silly.
    mapping = np.array([unk, unk, unk, unk, unk, unk, unk,  0,   1, unk,
                        unk,   2,   3,   4, unk, unk, unk,  5, unk,   6,
                          7,   8,   9,  10,  11,  12,  13, 14,  15, unk,
                        unk,  16,  17,  18, unk], dtype=np.int8)
    return mapping[np.minimum(im, len(mapping)-1)]  # clamping to map 255 to the last -1 up here.


def trainId2id(im, unk=0):
    # See `id2trainId` comment.
    mapping = np.array([7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33, unk], dtype=np.uint8)
    return mapping[np.minimum(im, len(mapping)-1)]  # clamping to map 255 to `unk` up here.
