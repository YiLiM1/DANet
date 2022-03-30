import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from dataloader.data_transform import *
import scipy.io as sio


class iBims1Dataset(Dataset):
    """iBims-1 Dataset."""

    def __init__(self, imagelist, transform=None):
        with open(imagelist) as f:
            image_names=f.readlines()
        self.image_names=[x.strip() for x in image_names]
        self.transform=transform

    def __getitem__(self, idx):
        image_name=self.image_names[idx]
        image_data=sio.loadmat('./data/iBims1/ibims1_core_mat/'+image_name)
        data=image_data['data']

        image=data['rgb'][0][0]  # RGB image
        depth=data['depth'][0][0]  # Raw depth map
        edges=data['edges'][0][0]  # Ground truth edges
        calib=data['calib'][0][0]  # Calibration parameters
        mask_invalid=data['mask_invalid'][0][0]  # Mask for invalid pixels
        mask_transp=data['mask_transp'][0][0]  # Mask for transparent pixels

        mask_wall=data['mask_wall'][0][0]  # RGB image
        mask_wall_paras=data['mask_wall_paras'][0][0]  # Raw depth map
        mask_table=data['mask_table'][0][0]  # Ground truth edges
        mask_table_paras=data['mask_table_paras'][0][0]  # Calibration parameters
        mask_floor=data['mask_floor'][0][0]  # Mask for invalid pixels
        mask_floor_paras=data['mask_floor_paras'][0][0]

        # print(image_name,mask_wall_paras)
        image=Image.fromarray(image)
        depth=Image.fromarray(depth)
        edges=Image.fromarray(edges)
        calib=Image.fromarray(calib)
        mask_invalid=Image.fromarray(mask_invalid)
        mask_transp=Image.fromarray(mask_transp)
        mask_wall=Image.fromarray(mask_wall)
        mask_table=Image.fromarray(mask_table)
        mask_floor=Image.fromarray(mask_floor)

        sample={'image': image, 'depth': depth, 'edges': edges, 'calib': calib,
                'mask_invalid': mask_invalid, 'mask_transp': mask_transp, "mask_wall": mask_wall,
                "mask_wall_paras": mask_wall_paras, "mask_table": mask_table, "mask_table_paras": mask_table_paras,
                "mask_floor": mask_floor, "mask_floor_paras": mask_floor_paras}
        if self.transform:
            sample=self.transform(sample)
        return sample

    def __len__(self):
        return len(self.image_names)


class NYUDV2Dataset(Dataset):
    """NYUV2D dataset."""
    def __init__(self, csv_file, root_path, transform=None):
        self.frame=pd.read_csv(csv_file, header=None)
        self.transform=transform
        self.root_path=root_path

    def __getitem__(self, idx):
        image_name=self.frame.iloc[idx, 0]
        depth_name=self.frame.iloc[idx, 1]

        image=Image.open(self.root_path+image_name)
        depth=Image.open(self.root_path+depth_name)
        sample={'image': image, 'depth': depth}

        if self.transform:
            sample=self.transform(sample)
        return sample

    def __len__(self):
        return len(self.frame)

def getTrainingData_NYUDV2(batch_size, trainlist_path, root_path):
    __imagenet_pca = {
        'eigval': torch.Tensor([0.2175, 0.0188, 0.0045]),
        'eigvec': torch.Tensor([
            [-0.5675,  0.7192,  0.4009],
            [-0.5808, -0.0045, -0.8140],
            [-0.5836, -0.6948,  0.4203],
        ])
    }
    __imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                        'std': [0.229, 0.224, 0.225]}
    transformed_training = NYUDV2Dataset(csv_file=trainlist_path,
                                        root_path = root_path,
                                        transform=transforms.Compose([
                                            Scale(240),
                                            RandomHorizontalFlip(),
                                            RandomRotate(5),
                                            CenterCrop([304, 228], [152, 114]),
                                            ToTensor(),
                                            Lighting(0.1, __imagenet_pca[
                                                'eigval'], __imagenet_pca['eigvec']),
                                            ColorJitter(
                                                brightness=0.4,
                                                contrast=0.4,
                                                saturation=0.4,
                                            ),
                                            Normalize(__imagenet_stats['mean'],
                                                      __imagenet_stats['std'])
                                        ]))
    dataloader_training = DataLoader(transformed_training, batch_size,
                                     shuffle=True, num_workers=16, pin_memory=False)

    return dataloader_training

def getTestingData_NYUDV2(batch_size, testlist_path, root_path):

    __imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                        'std': [0.229, 0.224, 0.225]}

    transformed_testing = NYUDV2Dataset(csv_file=testlist_path,
                                        root_path=root_path,
                                        transform=transforms.Compose([
                                           Scale(240),
                                           CenterCrop([304, 228], [304, 228]),
                                           ToTensor(is_test=True),
                                           Normalize(__imagenet_stats['mean'],
                                                     __imagenet_stats['std'])
                                       ]))
    dataloader_testing = DataLoader(transformed_testing, batch_size,
                                    shuffle=False, num_workers=4, pin_memory=False)

    return dataloader_testing
def getTestingData_iBims1(batch_size=64):

    __imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                        'std': [0.229, 0.224, 0.225]}
    transformed_testing = iBims1Dataset(imagelist='./data/iBims1/imagelist.txt',
                                        transform=transforms.Compose([
                                        Scale_iBims1(240),
                                        CenterCrop_iBims1([304, 228], [152, 114]),
                                        ToTensor_iBims1(is_test=True),
                                        Normalize_iBims1(__imagenet_stats['mean'],
                                                        __imagenet_stats['std'])
                                        ]))
    dataloader_testing = DataLoader(transformed_testing, batch_size,
                                    shuffle=False, num_workers=0, pin_memory=False)

    return dataloader_testing
