# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Streaming images and labels from datasets created with dataset_tool.py."""

import os
import numpy as np
import zipfile
import PIL.Image
import json
import torch
import dnnlib
import yaml
try:
    import pyspng
except ImportError:
    pyspng = None

#----------------------------------------------------------------------------
# Abstract base class for datasets.

class Dataset(torch.utils.data.Dataset):
    def __init__(self,
        name,                   # Name of the dataset.
        raw_shape,              # Shape of the raw image data (NCHW).
        max_size    = None,     # Artificially limit the size of the dataset. None = no limit. Applied before xflip.
        use_labels  = False,    # Enable conditioning labels? False = label dimension is zero.
        xflip       = False,    # Artificially double the size of the dataset via x-flips. Applied after max_size.
        random_seed = 0,        # Random seed to use when applying max_size.
        cache       = False,    # Cache images in CPU memory?
    ):
        self._name = name
        self._raw_shape = list(raw_shape)
        self._use_labels = use_labels
        self._cache = cache
        self._cached_images = dict() # {raw_idx: np.ndarray, ...}
        self._raw_labels = None
        self._label_shape = None

        # Apply max_size.
        self._raw_idx = np.arange(self._raw_shape[0], dtype=np.int64)
        if (max_size is not None) and (self._raw_idx.size > max_size):
            np.random.RandomState(random_seed % (1 << 31)).shuffle(self._raw_idx)
            self._raw_idx = np.sort(self._raw_idx[:max_size])

        # Apply xflip.
        self._xflip = np.zeros(self._raw_idx.size, dtype=np.float64)
        if xflip:
            self._raw_idx = np.tile(self._raw_idx, 2)
            self._xflip = np.concatenate([self._xflip, np.ones_like(self._xflip)])

    def _get_raw_labels(self):
        if self._raw_labels is None:
            self._raw_labels = self._load_raw_labels() if self._use_labels else None
            if self._raw_labels is None:
                self._raw_labels = np.zeros([self._raw_shape[0], 0], dtype=np.float32)
            assert isinstance(self._raw_labels, np.ndarray)
            assert self._raw_labels.shape[0] == self._raw_shape[0]
            assert self._raw_labels.dtype in [np.float32, np.int64]
            if self._raw_labels.dtype == np.int64:
                assert self._raw_labels.ndim == 1
                assert np.all(self._raw_labels >= 0)
        return self._raw_labels

    def close(self): # to be overridden by subclass
        pass

    def _load_raw_image(self, raw_idx): # to be overridden by subclass
        raise NotImplementedError

    def _load_raw_labels(self): # to be overridden by subclass
        raise NotImplementedError

    def __getstate__(self):
        return dict(self.__dict__, _raw_labels=None)

    def __del__(self):
        try:
            self.close()
        except:
            pass

    def __len__(self):
        return self._raw_idx.size

    def __getitem__(self, idx):
        raw_idx = self._raw_idx[idx]
        image, label = self._load_raw_image(raw_idx)

        assert isinstance(image, np.ndarray)
        assert list(image.shape) == self.image_shape
        assert image.dtype == np.float64

        return image.copy(), label.copy()


    def get_details(self, idx):
        d = dnnlib.EasyDict()
        d.raw_idx = int(self._raw_idx[idx])
        d.xflip = (int(self._xflip[idx]) != 0)
        d.raw_label = self._get_raw_labels()[d.raw_idx].copy()
        return d

    @property
    def name(self):
        return self._name

    @property
    def image_shape(self):
        return list(self._raw_shape[1:])

    @property
    def num_channels(self):
        assert len(self.image_shape) == 3 # CHW
        return self.image_shape[0]

    @property
    def resolution(self):
        assert len(self.image_shape) == 3 # CHW
        assert self.image_shape[1] == self.image_shape[2]
        return self.image_shape[1]

    @property
    def label_shape(self):
        if self._label_shape is None:
            raw_labels = self._get_raw_labels()
            if raw_labels.dtype == np.int64:
                self._label_shape = [int(np.max(raw_labels)) + 1]
            else:
                self._label_shape = raw_labels.shape[1:]
        return list(self._label_shape)

    @property
    def label_dim(self):
        assert len(self.label_shape) == 1
        return self.label_shape[0]

    @property
    def has_labels(self):
        return any(x != 0 for x in self.label_shape)

    @property
    def has_onehot_labels(self):
        return self._get_raw_labels().dtype == np.int64

#----------------------------------------------------------------------------
# Dataset subclass that loads images recursively from the specified directory
# or ZIP file.

class ImageFolderDataset(Dataset):
    def __init__(self,
        path,                   # Path to directory or zip.
        stats_path,
        exp         = None,     # Type of experiment
        num_classes = None,     # Number of classes
        resolution      = None, # Ensure specific resolution, None = highest available.
        use_pyspng      = True, # Use pyspng if available?
        **super_kwargs,         # Additional arguments for the Dataset base class.
    ):
        self._path = path
        self._stats_path = stats_path
        self._exp = exp
        self._num_classes = num_classes
        self._use_pyspng = use_pyspng
        self._zipfile = None

        # load stats
        with open(self._stats_path, "r") as f:
            self.Stats = yaml.safe_load(f)


        if os.path.isdir(self._path):
            self._type = 'dir'
            self._all_fnames = {os.path.relpath(os.path.join(root, fname), start=self._path) for root, _dirs, files in os.walk(self._path) for fname in files}
        elif self._file_ext(self._path) == '.zip':
            self._type = 'zip'
            self._all_fnames = set(self._get_zipfile().namelist())
        else:
            raise IOError('Path must point to a directory or zip')

        PIL.Image.init()
        self._image_fnames = sorted(fname for fname in self._all_fnames if self._file_ext(fname) in PIL.Image.EXTENSION or self._file_ext(fname) == '.npz')
        if len(self._image_fnames) == 0:
            raise IOError('No image files found in the specified path')

        name = os.path.splitext(os.path.basename(self._path))[0]
        image_temp, label_temp = self._load_raw_image(0)
        raw_shape = [len(self._image_fnames)] + list(image_temp.shape)
        if resolution is not None and (raw_shape[2] != resolution or raw_shape[3] != resolution):
            raise IOError('Image files do not match the specified resolution')
        super().__init__(name=name, raw_shape=raw_shape, **super_kwargs)
        

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def _get_zipfile(self):
        assert self._type == 'zip'
        if self._zipfile is None:
            self._zipfile = zipfile.ZipFile(self._path)
        return self._zipfile

    def _open_file(self, fname):
        if self._type == 'dir':
            return open(os.path.join(self._path, fname), 'rb')
        if self._type == 'zip':
            return self._get_zipfile().open(fname, 'r')
        return None

    def close(self):
        try:
            if self._zipfile is not None:
                self._zipfile.close()
        finally:
            self._zipfile = None

    def __getstate__(self):
        return dict(super().__getstate__(), _zipfile=None)

    def _load_raw_image(self, raw_idx):
        fname = self._image_fnames[raw_idx]
        image_normalized, label = self.get_normalized_image(fname)
        return image_normalized, label

    def get_normalized_image(self, fname):

        if self._exp == 'fixed_params':
            with self._open_file(fname) as f:
                data = np.load(f)

                last_value = data['last_value'].reshape(64,64)
                init_value = data['init_value'].reshape(64,64)

                init_value = float(self.Stats[self._exp]['u']['a']) * init_value + float(self.Stats[self._exp]['u']['b'])
                last_value = float(self.Stats[self._exp]['v']['a']) * last_value + float(self.Stats[self._exp]['v']['b'])

                image = np.stack((init_value, last_value), axis=-1)
                image = image.astype(np.float64).transpose(2, 0, 1) # HWX => CHW

                label = data['label'].astype(np.int64)
                onehot = np.zeros(self._num_classes, dtype=np.float32)
                onehot[label] = 1.0
                label = onehot
                return image, label

        elif self._exp == 'var_params':
            with self._open_file(fname) as f:
                data = np.load(f)

                label =data['label']
                last_value = data['last_value'].reshape(64,64)
                init_value = data['init_value'].reshape(64,64)
                init_value = float(self.Stats[self._exp]['u']['a']) * init_value + float(self.Stats[self._exp]['u']['b'])
                last_value = float(self.Stats[self._exp]['v']['a']) * last_value + float(self.Stats[self._exp]['v']['b'])

                if label == 0:
                    param = float(self.Stats[self._exp]['p']['diffusion']['a']) * data['diffusivity'] + float(self.Stats[self._exp]['p']['diffusion']['b'])
                elif label == 1:
                    param = float(self.Stats[self._exp]['p']['advection']['a']) * data['velocity_x'] + float(self.Stats[self._exp]['p']['advection']['b'])
                elif label == 2:
                    param = float(self.Stats[self._exp]['p']['advection_diffusion']['a']) * data['diffusivity'] + float(self.Stats[self._exp]['p']['advection_diffusion']['b'])

                image = np.stack((param*np.ones(init_value.shape), init_value, last_value), axis=-1)
                image = image.astype(np.float64).transpose(2, 0, 1) # HWX => CHW

                label = label.astype(np.int64)
                onehot = np.zeros(self._num_classes, dtype=np.float32)
                onehot[label] = 1.0
                label = onehot
                return image, label

        elif self._exp == 'hetro_params':
            with self._open_file(fname) as f:
                data = np.load(f)

                label =data['label']
                last_value = data['last_value'].reshape(64,64)
                init_value = data['init_value'].reshape(64,64)
                init_value = float(self.Stats[self._exp]['init']['a']) * init_value + float(self.Stats[self._exp]['init']['b'])
                last_value = float(self.Stats[self._exp]['last']['a']) * last_value + float(self.Stats[self._exp]['last']['b'])

                if label == 0:
                    diffusivity = float(self.Stats[self._exp]['p']['diffusion']['diffusivity']['a']) * data['diffusivity'] + float(self.Stats[self._exp]['p']['diffusion']['diffusivity']['b'])
                    param =diffusivity * np.ones(init_value.shape)

                    image = np.stack((param, init_value, last_value), axis=-1)

                elif label == 1:
                    velocity_x = float(self.Stats[self._exp]['p']['advection']['velocity_x']['a']) * data['velocity_x'] + float(self.Stats[self._exp]['p']['advection']['velocity_x']['b'])
                    velocity_y = float(self.Stats[self._exp]['p']['advection']['velocity_y']['a']) * data['velocity_y'] + float(self.Stats[self._exp]['p']['advection']['velocity_y']['b'])
                    param =np.ones(init_value.shape)
                    param[0:32] = velocity_x
                    param[32:64] = velocity_y                
                    image = np.stack((param, init_value, last_value), axis=-1)

                elif label == 2:
                    diffusivity = float(self.Stats[self._exp]['p']['advection_diffusion']['diffusivity']['a']) * data['diffusivity'] + float(self.Stats[self._exp]['p']['advection_diffusion']['diffusivity']['b'])
                    velocity_x = float(self.Stats[self._exp]['p']['advection_diffusion']['velocity_x']['a']) * data['velocity_x'] + float(self.Stats[self._exp]['p']['advection_diffusion']['velocity_x']['b'])
                    velocity_y = float(self.Stats[self._exp]['p']['advection_diffusion']['velocity_y']['a']) * data['velocity_y'] + float(self.Stats[self._exp]['p']['advection_diffusion']['velocity_y']['b'])
                    param =np.ones(init_value.shape)
                    param[0:21] = velocity_x
                    param[21:42] = velocity_y
                    param[42:64] = diffusivity
                    image = np.stack((param, init_value, last_value), axis=-1)

                image = image.astype(np.float64).transpose(2, 0, 1) # HWX => CHW

                label = label.astype(np.int64)
                onehot = np.zeros(self._num_classes, dtype=np.float32)
                onehot[label] = 1.0
                label = onehot
                return image, label


        elif self._exp == 'hetro_params_physics':
            with self._open_file(fname) as f:
                data = np.load(f)

                label =data['label']
                last_value = data['last_value'].reshape(64,64)
                init_value = data['init_value'].reshape(64,64)
                init_value = float(self.Stats[self._exp]['u']['a']) * init_value + float(self.Stats[self._exp]['u']['b'])
                last_value = float(self.Stats[self._exp]['v']['a']) * last_value + float(self.Stats[self._exp]['v']['b'])

                if label == 0:
                    diffusivity = float(self.Stats[self._exp]['p']['diffusion']['diffusivity']['a']) * data['diffusivity'] + float(self.Stats[self._exp]['p']['diffusion']['diffusivity']['b'])
                    param =np.zeros(init_value.shape)
                    param[42:64] = diffusivity

                    image = np.stack((param, init_value, last_value), axis=-1)

                elif label == 1:
                    velocity_x = float(self.Stats[self._exp]['p']['advection']['velocity_x']['a']) * data['velocity_x'] + float(self.Stats[self._exp]['p']['advection']['velocity_x']['b'])
                    velocity_y = float(self.Stats[self._exp]['p']['advection']['velocity_y']['a']) * data['velocity_y'] + float(self.Stats[self._exp]['p']['advection']['velocity_y']['b'])
                    param =np.zeros(init_value.shape)
                    param[0:21] = velocity_x
                    param[21:42] = velocity_y                
                    image = np.stack((param, init_value, last_value), axis=-1)

                elif label == 2:
                    diffusivity = float(self.Stats[self._exp]['p']['advection_diffusion']['diffusivity']['a']) * data['diffusivity'] + float(self.Stats[self._exp]['p']['advection_diffusion']['diffusivity']['b'])
                    velocity_x = float(self.Stats[self._exp]['p']['advection_diffusion']['velocity_x']['a']) * data['velocity_x'] + float(self.Stats[self._exp]['p']['advection_diffusion']['velocity_x']['b'])
                    velocity_y = float(self.Stats[self._exp]['p']['advection_diffusion']['velocity_y']['a']) * data['velocity_y'] + float(self.Stats[self._exp]['p']['advection_diffusion']['velocity_y']['b'])
                    param =np.zeros(init_value.shape)
                    param[0:21] = velocity_x
                    param[21:42] = velocity_y
                    param[42:64] = diffusivity
                    image = np.stack((param, init_value, last_value), axis=-1)

                image = image.astype(np.float64).transpose(2, 0, 1) # HWX => CHW

                label = label.astype(np.int64)
                onehot = np.zeros(self._num_classes, dtype=np.float32)
                onehot[label] = 1.0
                label = onehot


                return image, label

        elif self._exp == 'scalar_vector':
            with self._open_file(fname) as f:
                data = np.load(f)

                label =data['label']

                if label in [0, 1, 2]:
                    last_value = data['last_value'].reshape(64,64)
                    init_value = data['init_value'].reshape(64,64)
                    init_value = float(self.Stats[self._exp]['adve_diff']['init']['a']) * init_value + float(self.Stats[self._exp]['adve_diff']['init']['b'])
                    last_value = float(self.Stats[self._exp]['adve_diff']['last']['a']) * last_value + float(self.Stats[self._exp]['adve_diff']['last']['b'])

                    if label == 0:
                        param = float(self.Stats[self._exp]['adve_diff']['p']['diffusion']['a']) * data['diffusivity'] + float(self.Stats[self._exp]['adve_diff']['p']['diffusion']['b'])
                    elif label == 1:
                        param = float(self.Stats[self._exp]['adve_diff']['p']['advection']['a']) * data['velocity_x'] + float(self.Stats[self._exp]['adve_diff']['p']['advection']['b'])
                    elif label == 2:
                        param = float(self.Stats[self._exp]['adve_diff']['p']['advection_diffusion']['a']) * data['diffusivity'] + float(self.Stats[self._exp]['adve_diff']['p']['advection_diffusion']['b'])

                    image = np.stack((param*np.ones(init_value.shape), init_value, last_value), axis=-1)

                elif label == 5:
                    last_value = data['last_value_u'].reshape(64,64)
                    init_value = data['init_value_u'].reshape(64,64)
                    init_value = float(self.Stats[self._exp]['allen_cahn']['init']['a']) * init_value + float(self.Stats[self._exp]['allen_cahn']['init']['b'])
                    last_value = float(self.Stats[self._exp]['allen_cahn']['last']['a']) * last_value + float(self.Stats[self._exp]['allen_cahn']['last']['b'])
                    param = float(self.Stats[self._exp]['allen_cahn']['p']['diffusivity']['a']) * data['diffusivity'] + float(self.Stats[self._exp]['allen_cahn']['p']['diffusivity']['b'])
                    image = np.stack((param*np.ones(init_value.shape), init_value, last_value), axis=-1)    

                elif label in [3, 4]:
                    init_value_u = data['init_value_u'].reshape(64,64)
                    init_value_v = data['init_value_v'].reshape(64,64)
                    if label == 3:
                        last_value =  data['last_value_u'].reshape(64,64)
                    elif label == 4:
                        last_value =  data['last_value_v'].reshape(64,64)
                    init_value_u = float(self.Stats[self._exp]['burgers']['init']['a']) * init_value_u + float(self.Stats[self._exp]['burgers']['init']['b'])
                    init_value_v = float(self.Stats[self._exp]['burgers']['init']['a']) * init_value_v + float(self.Stats[self._exp]['burgers']['init']['b'])
                    last_value_u = float(self.Stats[self._exp]['burgers']['last']['a']) * last_value + float(self.Stats[self._exp]['burgers']['last']['b'])
                    image = np.stack((init_value_u, init_value_v, last_value_u), axis=-1)    


                elif label in [6, 7, 8]:
                    init_value_u = data['init_value_u'].reshape(64,64)
                    init_value_v = data['init_value_v'].reshape(64,64)

                    init_value_u = float(self.Stats[self._exp]['Navier_Stokes']['init']['a']) * init_value_u + float(self.Stats[self._exp]['Navier_Stokes']['init']['b'])
                    init_value_v = float(self.Stats[self._exp]['Navier_Stokes']['init']['a']) * init_value_v + float(self.Stats[self._exp]['Navier_Stokes']['init']['b'])

                    if label == 6:
                        last_value =  data['last_value_u'].reshape(64,64)
                        last_value = float(self.Stats[self._exp]['Navier_Stokes']['last_u']['a']) * last_value + float(self.Stats[self._exp]['Navier_Stokes']['last_u']['b'])
                    elif label == 7:
                        last_value =  data['last_value_v'].reshape(64,64)
                        last_value = float(self.Stats[self._exp]['Navier_Stokes']['last_v']['a']) * last_value + float(self.Stats[self._exp]['Navier_Stokes']['last_v']['b'])
                    elif label == 8:
                        error
                        # last_value =  data['last_value_p'].reshape(64,64)
                        # last_value = float(self.Stats[self._exp]['Navier_Stokes']['last_p']['a']) * last_value + float(self.Stats[self._exp]['Navier_Stokes']['last_p']['b'])
                    image = np.stack((init_value_u, init_value_v, last_value), axis=-1)    

                image = image.astype(np.float64).transpose(2, 0, 1) # HWX => CHW
                label = label.astype(np.int64)
                onehot = np.zeros(self._num_classes, dtype=np.float32)
                onehot[label] = 1.0
                label = onehot
                return image, label

        else:
            raise ValueError(f"Invalid experiment: {self._exp}")


#----------------------------------------------------------------------------
