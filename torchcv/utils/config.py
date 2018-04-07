from pprint import  pprint
class Config:
    lr = 0.1
    resume = False
    checkpoint = '/home/claude.cy/file/ssd/checkpoint/'
    data_root = '/home/claude.cy/.data/all_images'
    voc07_trainval = 'torchcv/datasets/voc/voc07_trainval.txt'
    voc12_trainval = 'torchcv/datasets/voc/voc12_trainval.txt'
    voc07_test = 'torchcv/datasets/voc/voc07_test.txt'
    batch_size = 32
    num_worker = 8
    plot_every = 20
    debug_file = '/tmp/debugdsod'
    load_path = None #'examples/ssd/checkpoint/ckpt.pth'
    img_size = 300
    env = 'dsod'
    iter_size = 4
    eval_every=5

    def _parse(self, kwargs):
        state_dict = self._state_dict()
        for k, v in kwargs.items():
            if k not in state_dict:
                raise ValueError('UnKnown Option: "--%s"' % k)
            setattr(self, k, v)

        print('======user config========')
        pprint(self._state_dict())
        print('==========end============')

    def _state_dict(self):
        return {k: getattr(self, k) for k, _ in Config.__dict__.items() \
                if not k.startswith('_')}


opt = Config()
