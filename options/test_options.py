from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--ntest', type=int, default=float("inf"), help='# of test examples.')
        self.parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        self.parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
        self.parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        self.parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument('--how_many', type=int, default=200, help='how many test images to run')
        self.parser.add_argument('--map_name', type=str, default='uv_seg', help='mapping function')
        self.parser.add_argument('--part_info', type=str, default='assets/pretrains/smpl_part_info.json',
                                  help='smpl part info path.')
        self.parser.add_argument('--uv_mapping', type=str, default='assets/pretrains/mapper.txt',
                                  help='uv mapping.')
        self.parser.add_argument('--hmr_model', type=str, default='assets/pretrains/hmr_tf2pt.pth',
                                  help='pretrained hmr model path.')
        self.parser.add_argument('--smpl_model', type=str, default='assets/pretrains/smpl_model.pkl',
                                  help='pretrained smpl model path.')

        self.isTrain = False
