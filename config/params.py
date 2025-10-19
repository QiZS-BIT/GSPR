import os


class ModelParams:
    def __init__(self):
        self.base_root = "/media/octane17/T7ShieldNus/GSPR/data"
        self.project_root = "/home/octane17/GSPR"
        # BS split
        self.info_root = os.path.join(self.base_root, "nuscenes_infos-bs.pkl")
        self.gaussian_dataset_root = os.path.join(self.base_root, "NuscenesGaussianModel/BS_4096_VFE")
        self.database_index_list = [os.path.join(self.base_root, "bs_db.npy")]
        self.train_query_index_list = [os.path.join(self.base_root, "bs_db.npy"),
                                       os.path.join(self.base_root, "bs_train_query.npy")]
        self.val_query_index_list = [os.path.join(self.base_root, "bs_test_query.npy")]
        self.test_query_index_list = [os.path.join(self.base_root, "bs_test_query.npy")]
        # SON split
        self.info_root = os.path.join(self.base_root, "nuscenes_infos-son.pkl")
        self.gaussian_dataset_root = os.path.join(self.base_root, "NuscenesGaussianModel/SON_4096_VFE")
        self.database_index_list = [os.path.join(self.base_root, "son_db.npy")]
        self.test_query_index_list = [os.path.join(self.base_root, "son_test_query.npy")]
        # SQ split
        self.info_root = os.path.join(self.base_root, "nuscenes_infos-sq.pkl")
        self.gaussian_dataset_root = os.path.join(self.base_root, "NuscenesGaussianModel/SQ_4096_VFE")
        self.database_index_list = [os.path.join(self.base_root, "sq_db.npy")]
        self.test_query_index_list = [os.path.join(self.base_root, "sq_test_query.npy")]

        self.checkpoint_path = "/home/octane17/GSPR/checkpoint/GSPR.pth.tar"
        self.training_root = "/home/octane17/GSPR/runs"
        if not os.path.exists(self.training_root):
            os.makedirs(self.training_root)
        self.weights_root = os.path.join(self.training_root, "weights")
        if not os.path.exists(self.weights_root):
            os.makedirs(self.weights_root)
        self.logs_root = os.path.join(self.training_root, "logs")
        if not os.path.exists(self.logs_root):
            os.makedirs(self.logs_root)
        self.cache_root = os.path.join(self.training_root, "cache")
        if not os.path.exists(self.cache_root):
            os.makedirs(self.cache_root)
        self.resume_checkpoint = True
        self.resume_epoch = 10
        self.epochs = 10
        self.learning_rate = 0.00001
        self.pos_num = 2
        self.neg_num = 6
        self.pos_dist_threshold = 9
        self.neg_dist_threshold = 18
        self.gth_dist_threshold = 18
        self.loss_margin = 0.5
        self.num_workers = 1

        self.output_dim = 256
