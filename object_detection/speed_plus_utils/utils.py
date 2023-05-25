import numpy as np
import json
import os
from PIL import Image
from matplotlib import pyplot as plt

# deep learning framework imports
try:
    from tensorflow.keras.utils import Sequence
    from tensorflow.keras.preprocessing import image as keras_image

    has_tf = True
except ModuleNotFoundError:
    has_tf = False

try:
    import torch
    from torch.utils.data import Dataset
    from torchvision import transforms

    has_pytorch = True
except ImportError:
    has_pytorch = False


class Camera:
    """" Utility class for accessing camera parameters. """

    speed_root = 'speed_plus_utils'

    with open(os.path.join(speed_root, 'camera.json'), 'r') as f:
        camera_params = json.load(f)

    fx = camera_params['fx']  # focal length[m]
    fy = camera_params['fy']  # focal length[m]
    nu = camera_params['Nu']  # number of horizontal[pixels]
    nv = camera_params['Nv']  # number of vertical[pixels]
    ppx = camera_params['ppx']  # horizontal pixel pitch[m / pixel]
    ppy = camera_params['ppy']  # vertical pixel pitch[m / pixel]
    fpx = fx / ppx  # horizontal focal length[pixels]
    fpy = fy / ppy  # vertical focal length[pixels]
    k = camera_params['cameraMatrix']
    K = np.array(k)  # cameraMatrix
    dcoef = camera_params['distCoeffs']


def process_json_dataset(root_dir):
    with open(os.path.join(root_dir, 'synthetic', 'train.json'), 'r') as f:
        train_images_labels = json.load(f)

    with open(os.path.join(root_dir, 'synthetic', 'validation.json'), 'r') as f:
        test_image_list = json.load(f)

    with open(os.path.join(root_dir, 'sunlamp', 'test.json'), 'r') as f:
        sunlamp_image_list = json.load(f)

    with open(os.path.join(root_dir, 'lightbox', 'test.json'), 'r') as f:
        lightbox_image_list = json.load(f)

    partitions = {'validation': [], 'train': [], 'sunlamp': [], 'lightbox': []}
    labels = {}

    for image_ann in train_images_labels:
        partitions['train'].append(image_ann['filename'])
        labels[image_ann['filename']] = {'q': image_ann['q_vbs2tango_true'], 'r': image_ann['r_Vo2To_vbs_true']}

    for image in test_image_list:
        partitions['validation'].append(image['filename'])

    for image in sunlamp_image_list:
        partitions['sunlamp'].append(image['filename'])

    for image in lightbox_image_list:
        partitions['lightbox'].append(image['filename'])

    return partitions, labels


def quat2dcm(q):
    """ Computing direction cosine matrix from quaternion, adapted from PyNav. """

    # normalizing quaternion
    q = q / np.linalg.norm(q)

    q0 = q[0]
    q1 = q[1]
    q2 = q[2]
    q3 = q[3]

    dcm = np.zeros((3, 3))

    dcm[0, 0] = 2 * q0 ** 2 - 1 + 2 * q1 ** 2
    dcm[1, 1] = 2 * q0 ** 2 - 1 + 2 * q2 ** 2
    dcm[2, 2] = 2 * q0 ** 2 - 1 + 2 * q3 ** 2

    dcm[0, 1] = 2 * q1 * q2 + 2 * q0 * q3
    dcm[0, 2] = 2 * q1 * q3 - 2 * q0 * q2

    dcm[1, 0] = 2 * q1 * q2 - 2 * q0 * q3
    dcm[1, 2] = 2 * q2 * q3 + 2 * q0 * q1

    dcm[2, 0] = 2 * q1 * q3 + 2 * q0 * q2
    dcm[2, 1] = 2 * q2 * q3 - 2 * q0 * q1

    return dcm


def project(q, r, points = np.array([[0, 0, 0],
                                     [1, 0, 0],
                                     [0, 1, 0],
                                     [0, 0, 1]])):
    """ Projecting points to image frame to draw axes """

    # reference points in satellite frame for drawing axes
    p_axes = np.column_stack((points, np.ones(points.shape[0])))
    points_body = np.transpose(p_axes)

    # transformation to camera frame
    pose_mat = np.hstack((np.transpose(quat2dcm(q)), np.expand_dims(r, 1)))
    p_cam = np.dot(pose_mat, points_body)

    # getting homogeneous coordinates
    points_camera_frame = p_cam / p_cam[2]

    x0, y0 = (points_camera_frame[0], points_camera_frame[1])

    # apply distortion
    dist = Camera.dcoef

    r2 = x0 * x0 + y0 * y0
    cdist = 1 + dist[0] * r2 + dist[1] * r2 * r2 + dist[4] * r2 * r2 * r2
    x1 = x0 * cdist + dist[2] * 2 * x0 * y0 + dist[3] * (r2 + 2 * x0 * x0)
    y1 = y0 * cdist + dist[2] * (r2 + 2 * y0 * y0) + dist[3] * 2 * x0 * y0

    # projection to image plane
    x = Camera.K[0, 0] * x1 + Camera.K[0, 2]
    y = Camera.K[1, 1] * y1 + Camera.K[1, 2]

    return np.column_stack((x, y))


class SatellitePoseEstimationDataset:
    """ Class for dataset inspection: easily accessing single images, and corresponding ground truth pose data. """

    def __init__(self, root_dir='datasets/'):
        self.partitions, self.labels = process_json_dataset(root_dir)
        self.root_dir = root_dir

    def get_image(self, i=0, split='train'):

        """ Loading image as PIL image. """

        img_name = self.partitions[split][i]
        if split == 'train':
            img_name = os.path.join(self.root_dir, 'synthetic', 'images', img_name)
        elif split == 'validation':
            img_name = os.path.join(self.root_dir, 'synthetic', 'images', img_name)
        elif split == 'sunlamp':
            img_name = os.path.join(self.root_dir, 'sunlamp', 'images', img_name)
        elif split == 'lightbox':
            img_name = os.path.join(self.root_dir, 'lightbox', 'images', img_name)
        else:
            print()
            # raise error?

        image = Image.open(img_name).convert('RGB')
        return image

    def get_pose(self, i=0):

        """ Getting pose label for image. """

        img_id = self.partitions['train'][i]
        q, r = self.labels[img_id]['q'], self.labels[img_id]['r']
        return q, r

    def visualize(self, i, partition='train', ax=None):

        """ Visualizing image, with ground truth pose with axes projected to training image. """

        if ax is None:
            ax = plt.gca()
        img = self.get_image(i)
        ax.imshow(img)

        # no pose label for test
        if partition == 'train':
            q, r = self.get_pose(i)
            points = project(q, r)
            xa = points[:,0]
            ya = points[:,1]
            ax.arrow(xa[0], ya[0], xa[1] - xa[0], ya[1] - ya[0], head_width=30, color='r')
            ax.arrow(xa[0], ya[0], xa[2] - xa[0], ya[2] - ya[0], head_width=30, color='g')
            ax.arrow(xa[0], ya[0], xa[3] - xa[0], ya[3] - ya[0], head_width=30, color='b')

        return


if has_pytorch:
    class PyTorchSatellitePoseEstimationDataset(Dataset):

        """ SPEED dataset that can be used with DataLoader for PyTorch training. """

        def __init__(self, split='train', speed_root='datasets/', transform=None):

            if not has_pytorch:
                raise ImportError('Pytorch was not imported successfully!')

            if split not in {'train', 'validation', 'sunlamp', 'lightbox'}:
                raise ValueError(
                    'Invalid split, has to be either \'train\', \'validation\', \'sunlamp\' or \'lightbox\'')

            if split in {'train', 'validation'}:
                self.image_root = os.path.join(speed_root, 'synthetic', 'images')
                with open(os.path.join(speed_root, "synthetic", split + '.json'), 'r') as f:
                    label_list = json.load(f)
            else:
                self.image_root = os.path.join(speed_root, split, 'images')
                with open(os.path.join(speed_root, split, 'test.json'), 'r') as f:
                    label_list = json.load(f)

            self.sample_ids = [label['filename'] for label in label_list]
            self.train = split == 'train'

            if self.train:
                self.labels = {label['filename']: {'q': label['q_vbs2tango_true'], 'r': label['r_Vo2To_vbs_true']}
                               for label in label_list}
            self.split = split
            self.transform = transform

        def __len__(self):
            return len(self.sample_ids)

        def __getitem__(self, idx):
            sample_id = self.sample_ids[idx]
            img_name = os.path.join(self.image_root, sample_id)

            # note: despite grayscale images, we are converting to 3 channels here,
            # since most pre-trained networks expect 3 channel input
            pil_image = Image.open(img_name).convert('RGB')

            if self.train:
                q, r = self.labels[sample_id]['q'], self.labels[sample_id]['r']
                y = np.concatenate([q, r])
            else:
                y = sample_id

            if self.transform is not None:
                torch_image = self.transform(pil_image)
            else:
                torch_image = pil_image

            return torch_image, y
else:
    class PyTorchSatellitePoseEstimationDataset:
        def __init__(self, *args, **kwargs):
            raise ImportError('Pytorch is not available!')

if has_tf:
    class KerasDataGenerator(Sequence):

        """ DataGenerator for Keras to be used with fit_generator (https://keras.io/models/sequential/#fit_generator)"""

        def __init__(self, preprocessor, label_list, speed_root, batch_size=32, dim=(224, 224), n_channels=3,
                     shuffle=True):

            # loading dataset
            self.image_root = os.path.join(speed_root, 'synthetic', 'images')

            # Initialization
            self.preprocessor = preprocessor
            self.dim = dim
            self.batch_size = batch_size
            self.labels = self.labels = {
                label['filename']: {'q': label['q_vbs2tango_true'], 'r': label['r_Vo2To_vbs_true']}
                for label in label_list}
            self.list_IDs = [label['filename'] for label in label_list]
            self.n_channels = n_channels
            self.shuffle = shuffle
            self.indexes = None
            self.on_epoch_end()

        def __len__(self):

            """ Denotes the number of batches per epoch. """

            return int(np.floor(len(self.list_IDs) / self.batch_size))

        def __getitem__(self, index):

            """ Generate one batch of data """

            # Generate indexes of the batch
            indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

            # Find list of IDs
            list_IDs_temp = [self.list_IDs[k] for k in indexes]

            # Generate data
            X, y = self.__data_generation(list_IDs_temp)

            return X, y

        def on_epoch_end(self):

            """ Updates indexes after each epoch """

            self.indexes = np.arange(len(self.list_IDs))
            if self.shuffle:
                np.random.shuffle(self.indexes)

        def __data_generation(self, list_IDs_temp):

            """ Generates data containing batch_size samples """

            # Initialization
            X = np.empty((self.batch_size, *self.dim, self.n_channels))
            y = np.empty((self.batch_size, 7), dtype=float)

            # Generate data
            for i, ID in enumerate(list_IDs_temp):
                img_path = os.path.join(self.image_root, ID)
                img = keras_image.load_img(img_path, target_size=(224, 224))
                x = keras_image.img_to_array(img)
                x = self.preprocessor(x)
                X[i,] = x

                q, r = self.labels[ID]['q'], self.labels[ID]['r']
                y[i] = np.concatenate([q, r])

            return X, y
else:
    class KerasDataGenerator:
        def __init__(self, *args, **kwargs):
            raise ImportError('tensorflow.keras is not available! Please install tensorflow.')