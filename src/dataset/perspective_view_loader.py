from src.dataset.preprocess import augmentor
from src.dataset.data_argument import *


class PerspectiveViewLoader:
    def __init__(self, dataset, config, data_len=-1, is_train=True, pcd_aug=False,
                 img_aug=False, use_padding=False):
        self.dataset = dataset
        self.config = config
        self.is_train = is_train
        self.pcd_aug = pcd_aug
        self.data_len = data_len
        self.img_aug = img_aug
        self.use_padding = use_padding
        self.feature_mean = np.array(config["sensor"]["img_mean"])
        self.feature_mean = self.feature_mean.reshape((1, self.feature_mean.shape[0], 1, 1))
        self.feature_std = np.array(config["sensor"]["img_stds"])
        self.feature_std = self.feature_std.reshape((1, self.feature_std.shape[0], 1, 1))
        if not self.is_train:
            self.pcd_aug = False
            self.img_aug = False
        augment_params = augmentor.AugmentParams()
        augment_config = self.config['augmentation']

        if self.pcd_aug:
            augment_params.setFlipProb(
                p_flipx=augment_config['p_flipx'], p_flipy=augment_config['p_flipy'])
            augment_params.setTranslationParams(
                p_transx=augment_config['p_transx'], trans_xmin=augment_config[
                    'trans_xmin'], trans_xmax=augment_config['trans_xmax'],
                p_transy=augment_config['p_transy'], trans_ymin=augment_config[
                    'trans_ymin'], trans_ymax=augment_config['trans_ymax'],
                p_transz=augment_config['p_transz'], trans_zmin=augment_config[
                    'trans_zmin'], trans_zmax=augment_config['trans_zmax'])
            augment_params.setRotationParams(
                p_rot_roll=augment_config['p_rot_roll'], rot_rollmin=augment_config[
                    'rot_rollmin'], rot_rollmax=augment_config['rot_rollmax'],
                p_rot_pitch=augment_config['p_rot_pitch'], rot_pitchmin=augment_config[
                    'rot_pitchmin'], rot_pitchmax=augment_config['rot_pitchmax'],
                p_rot_yaw=augment_config['p_rot_yaw'], rot_yawmin=augment_config[
                    'rot_yawmin'], rot_yawmax=augment_config['rot_yawmax'])
            self.augmentor = augmentor.Augmentor(augment_params)
        else:
            self.augmentor = None

        if self.img_aug:
            self.img_jitter = ImageColorJitter(
                *augment_config["img_jitter"])
        else:
            self.img_jitter = None

        projection_config = self.config['sensor']

        if self.use_padding:
            h_pad = projection_config["h_pad"]
            w_pad = projection_config["w_pad"]
            self.pad = ImagePad((w_pad, h_pad))
        else:
            h_pad = 0
            w_pad = 0

        if self.is_train:
            self.flip = ImageRandomHorizontalFlip(0.5)
            self.rotate = ImageRandomRotation(15)
            self.crop = ImageRandomCrop(size=(projection_config['proj_ht'] - 2 * h_pad,
                                              projection_config['proj_wt'] - 2 * w_pad))
        else:
            self.flip = None
            self.rotate = None
            self.crop = ImageCenterCrop(size=(projection_config['proj_h'] - 2 * h_pad,
                                              projection_config['proj_w'] - 2 * w_pad))

    def __getitem__(self, index):
        # feature: range, x, y, z, i, rgb
        pointcloud, sem_label, _ = self.dataset.loadDataByIndex(index)
        if self.pcd_aug:
            pointcloud = self.augmentor.doAugmentation(pointcloud)
        # get image feature
        image = self.dataset.loadImage(index)
        image = np.array(image)
        if self.img_aug:
            image = self.img_jitter.data_arg(image)
        seq_id, _ = self.dataset.parsePathInfoByIndex(index)
        mapped_pointcloud, keep_mask = self.dataset.mapLidar2Camera(
            seq_id, pointcloud[:, :3], image.shape[1], image.shape[0])

        y_data = mapped_pointcloud[:, 1].astype(np.int32)
        x_data = mapped_pointcloud[:, 0].astype(np.int32)

        # compute image view pointcloud feature
        image = image.astype(np.float32) / 255.0
        depth = np.linalg.norm(pointcloud[:, :3], 2, axis=1)
        keep_poincloud = pointcloud[keep_mask]
        proj_xyzi = np.zeros(
            (image.shape[0], image.shape[1], keep_poincloud.shape[1]), dtype=np.float32)
        proj_xyzi[x_data, y_data] = keep_poincloud
        proj_depth = np.zeros(
            (image.shape[0], image.shape[1]), dtype=np.float32)
        proj_depth[x_data, y_data] = depth[keep_mask]

        proj_label = np.zeros((image.shape[0], image.shape[1]), dtype=np.int32)

        try:
            proj_label[x_data, y_data] = self.dataset.labelMapping(sem_label[keep_mask])
        except Exception as msg:
            print(msg)
            print(keep_mask.shape)
            print(sem_label.shape)

        proj_mask = np.zeros(
            (image.shape[0], image.shape[1]), dtype=np.int32)
        proj_mask[x_data, y_data] = 1

        data_exp = np.concatenate((np.expand_dims(proj_depth, 0),
                                   proj_xyzi.transpose((2, 0, 1)),
                                   image.transpose((2, 0, 1)),
                                   np.expand_dims(proj_mask, 0),
                                   np.expand_dims(proj_label, 0)), 0)
        data_exp = data_exp.astype('float32')

        if self.is_train:
            data_exp = self.flip.data_arg(data_exp)
            # data_exp = self.rotate.data_arg(data_exp)
            data_exp = self.crop.data_arg(data_exp)
            if self.use_padding:
                data_exp = self.pad.data_arg(data_exp)
            input_feature = data_exp[:8]
            input_mask = data_exp[8]
            input_label = data_exp[9]
        else:
            data_exp = self.crop.data_arg(data_exp)
            if self.use_padding:
                data_exp = self.pad.data_arg(data_exp)
            input_feature = data_exp[:8]
            input_mask = data_exp[8]
            input_label = data_exp[9]

        input_feature[0:5] = (input_feature[0:5] - self.feature_mean) / self.feature_std * \
                                np.broadcast_to(np.expand_dims(input_mask, 0), input_feature[0:5].shape)
        pcd_feature = input_feature[0:5]
        img_feature = input_feature[5:8]
        input_label = input_label.astype("int32")
        label_mask = input_label > 0
        return pcd_feature, img_feature, label_mask, input_label

    def __len__(self):
        if self.data_len > 0 and self.data_len < len(self.dataset):
            return self.data_len
        else:
            return len(self.dataset)
