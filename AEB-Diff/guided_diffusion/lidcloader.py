import torch
import torch.nn
import os
import os.path
from skimage import io
import random

class LIDCDataset(torch.utils.data.Dataset):
    def __init__(self, directory, test_flag=True):
        '''
        directory is expected to contain some folder structure:
                  if some subfolder contains only files, all of these
                  files are assumed to have a name like
                  brats_train_001_XXX_123_w.nii.gz
                  where XXX is one of t1, t1ce, t2, flair, seg
                  we assume these five files belong to the same image
                  seg is supposed to contain the segmentation
        '''
        super().__init__()
        self.directory = os.path.expanduser(directory)

        self.test_flag=test_flag
        if test_flag:
            self.seqtypes = ['image', 'label0', 'label1', 'label2', 'label3']
        else:
            self.seqtypes = ['image', 'label0', 'label1', 'label2', 'label3']

        self.seqtypes_set = set(self.seqtypes)
        self.database = []
        for root, dirs, files in os.walk(self.directory):
            # if there are no subdirs, we have data
            if not dirs:
                files.sort()
                datapoint = dict()
                # extract all files as channels
                for f in files:
                    seqtype = f.split('_')[0]
                    #print(seqtype)
                    datapoint[seqtype] = os.path.join(root, f)
                    #print(datapoint)

                assert set(datapoint.keys()) == self.seqtypes_set, \
                    f'datapoint is incomplete, keys are {datapoint.keys()}'
                self.database.append(datapoint)


    def __getitem__(self, x):
        out = []
        filedict = self.database[x]
        for seqtype in self.seqtypes:
            img = io.imread(filedict[seqtype])
            img = img / 255
            #nib_img = nibabel.load(filedict[seqtype])
            path=filedict[seqtype]
            out.append(torch.tensor(img))
        out = torch.stack(out)
        if self.test_flag:
            image = out[0]
            image = torch.unsqueeze(image, 0)
            image = torch.cat((image,image,image,image), 0) #concatenating images 4 times is not necessary for LIDC dataset, but for MRI we concatenated all of them (flair, f1, f2, pd). This is for reference! :D
            labels = [out[i] for i in range(1, 5)]
            labels = torch.stack(labels, dim=0)
            return (image, labels, path)
        else:
            image = out[0]
            image = torch.unsqueeze(image, 0)
            image = torch.cat((image,image,image,image), 0)
            label = out[random.randint(1, 4)]
            label = torch.unsqueeze(label, 0)
            return (image, label)

    def __len__(self):
        return len(self.database)

class LIDCDatasetBeta(torch.utils.data.Dataset):
    def __init__(self, directory, test_flag=True):
        """
        加载包含 `image` 和 `label` 的数据集，支持概率值的 `label`。
        directory 是数据集主目录，test_flag 用于区分训练和测试模式。
        """
        super().__init__()
        self.directory = os.path.expanduser(directory)
        self.test_flag = test_flag
        self.seqtypes = ['image', 'label']  # 新的数据集包含 image 和 label
        self.database = []

        # 遍历目录，收集数据
        for root, dirs, files in os.walk(self.directory):
            if not dirs:
                files.sort()
                image_file = next((f for f in files if f.startswith('image') and f.endswith(('.jpg', '.png'))), None)
                label_file = next((f for f in files if f.startswith('label') and f.endswith(('.jpg', '.png'))), None)

                if image_file and label_file:
                    datapoint = {
                        'image': os.path.join(root, image_file),
                        'label': os.path.join(root, label_file)
                    }
                    self.database.append(datapoint)
                else:
                    print(f"跳过不完整数据点: {root}, 文件: {files}")

    def __getitem__(self, index):
        filedict = self.database[index]

        image = torch.tensor(io.imread(filedict['image']), dtype=torch.float32) / 255.0  # 归一化到 [0, 1]
        label = torch.tensor(io.imread(filedict['label']), dtype=torch.float32) / 255.0  # 概率值已归一化到 [0, 1]

        image = image.unsqueeze(0)  # 添加通道维度
        if not self.test_flag:
            label = label.unsqueeze(0)  # 添加通道维度
        image = torch.cat((image, image, image, image), dim=0)

        if self.test_flag:
            path = filedict['image']  # 返回图像路径
            return image, label, path
        else:
            return image, label
    def __len__(self):
        return len(self.database)
