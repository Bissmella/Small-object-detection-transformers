# YOLOv5 YOLO-specific modules
# with binary
import argparse
import logging
import sys
from copy import deepcopy
import scipy.io as sio
from torch import mode

sys.path.append('./')  # to run '$ python *.py' files in subdirectories
logger = logging.getLogger(__name__)

from .common import *
# from models.swin_transformer import *
from .experimental import *
from .image_encoder import *
# from models.edsr import EDSR
from ..utils.autoanchor import check_anchor_order
from ..utils.general import make_divisible, check_file, set_logging, xywh2xyxy
from ..utils.torch_utils import time_synchronized, fuse_conv_and_bn, model_info, scale_img, initialize_weights, \
    select_device, copy_attr
from codes.meta_SAM.segment_anything.segment_anything.build_sam import sam_model_registry
from codes.meta_SAM.segment_anything.segment_anything.automatic_mask_generator import SamAutomaticMaskGenerator
import torchvision.ops as ops
import scipy.io as sio
import numpy
# from models import build_model
try:
    import thop  # for FLOPS computation
except ImportError:
    thop = None
import torch.nn.functional as F


class Detect(nn.Module):
    stride = None  # strides computed during build
    export = False  # onnx export

    def __init__(self, nc=80, in_channels= 36864, representation_size=1024, anchors=(), ch=()):  # detection layer
        super(Detect, self).__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
 

        #for samDet
        self.fc6 = nn.Linear(in_channels, representation_size)
        self.fc7 = nn.Linear(representation_size, representation_size) 
            #prediction head
        self.cls_score = nn.Linear(representation_size, nc + 1)  # +1 should be added for background
        self.bbox_pred = nn.Linear(representation_size, 4) #num_classes * 4

    def forward(self, x):
        # x = x.copy()  # for profiling
        z = []  # inference output

        self.training |= self.export
        
        #for samDet
        x = x.flatten(start_dim=1)

        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))

        x = x.flatten(start_dim=1)
        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)
        z.append(scores)
        z.append(bbox_deltas)
        return torch.cat(z, 1) if self.training else (torch.cat(z, 1), x)

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()


class Model(nn.Module):
    export = False  # onnx export
    def __init__(self, cfg='yolov5s.yaml',input_mode='RGB',ch_steam=3, ch=3, nc=None, anchors=None,config=None,sr=False,factor=2):  #att=False,sr_att=False model, input channels, number of classes
        super(Model, self).__init__()
        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict
        else:  # is *.yaml
            import yaml  # for torch hub
            self.yaml_file = Path(cfg).name
            with open(cfg) as f:
                self.yaml = yaml.load(f, Loader=yaml.SafeLoader)  # model dict
        self.sr = sr
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # input channels
        if nc and nc != self.yaml['nc']:
            logger.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml['nc'] = nc  # override yaml value
        if anchors:
            logger.info(f'Overriding model.yaml anchors with anchors={anchors}')
            self.yaml['anchors'] = round(anchors)  # override yaml value
        # self.input_mode = input_mode
        if input_mode == 'RGB+IR+fusion':
            self.steam, _ = parse_model(deepcopy(self.yaml),'steam', ch=[ch_steam],config=config)  # zjq model, savelist
        #self.model, self.save = parse_model(deepcopy(self.yaml),'backbone+head', ch=[ch],config=config)  # model, savelist   #* changed removed
        model = "vit_b"
        weights_filepath = "/home/bbahaduri/sryolo/weights/sam_vit_b_01ec64.pth"
        model = sam_model_registry[model](checkpoint=weights_filepath)
        self.sam = model
        self.mask_generator = SamAutomaticMaskGenerator(
            model=self.sam,
            points_per_side=128,
            pred_iou_thresh=0.86,
            stability_score_thresh=0.92,
            crop_n_layers=0,
            crop_n_points_downscale_factor=1,
            #min_mask_region_area=100,  # Requires open-cv to run post-processing
            )

        self.detect = Detect(nc=8)
        
        # self.f1=self.yaml['f1']  #蒸馏特征层层数
        # self.f2=self.yaml['f2']
        # self.f3=self.yaml['f3']


        # Build strides, anchors
         # only run once
        # m = self.model[-2]  # Detect()
        # if isinstance(m, Detect):
        #     s = 256  # 2x min stride
        #     #m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch_steam, s, s),torch.zeros(1, ch_steam, s, s),input_mode)[0]])  # forward
        #     m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch_steam, s, s),torch.zeros(1, ch_steam, s, s),input_mode)[0]])  # forward
        #     m.anchors /= m.stride.view(-1, 1, 1)
        #     check_anchor_order(m)
        #     self.stride = m.stride
        #     self._initialize_biases()  # only run once
        # Init weights, biases
        initialize_weights(self)
        self.info()
        logger.info('')
        
    
    def forward(self, x, ir=torch.randn(1,3,512,512), input_mode='RGB+IR', augment=False, profile=False):

        # input_mode = 'RGB+IR' #IRRGB
        if input_mode=='RGB':
            ir=x
        if augment:
            img_size = x.shape[-2:]  # height, width
            s = [1, 0.83, 0.67]  # scales
            f = [None, 3, None]  # flips (2-ud, 3-lr)
            y = []  # outputs
            for si, fi in zip(s, f):
                xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
                iri = scale_img(ir.flip(fi) if fi else ir, si, gs=int(self.stride.max()))
                if input_mode =='RGB+IR+fusion':
                    steam1 = self.forward_once(x,'steam',profile)
                    steam2 = self.forward_once(ir,'steam',profile)
                    steam = torch.cat([steam1,steam2],1)
                if input_mode == 'RGB+IR':
                    steam = torch.cat([xi,iri[:,0:1,:,:]],1)
                if input_mode == 'RGB':
                    steam = xi
                if input_mode == 'IR':
                    steam = iri#steam = iri[:,0:1,:,:]
                if input_mode == 'RGB+IR+MF':
                    steam = [x,ir[:,0:1,:,:]] #[:,0:1,:,:]
                yi = self.forward_once(steam,'yolo')[0]  # forward
                # yi = self.forward_once(xi)[0]  # forward
                # cv2.imwrite('img%g.jpg' % s, 255 * xi[0].numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
                yi[..., :4] /= si  # de-scale
                if fi == 2:
                    yi[..., 1] = img_size[0] - yi[..., 1]  # de-flip ud
                elif fi == 3:
                    yi[..., 0] = img_size[1] - yi[..., 0]  # de-flip lr
                y.append(yi)
            return torch.cat(y, 1), None  # augmented inference, train
        else:
            if input_mode =='RGB+IR+fusion':
                steam1 = self.forward_once(x,'steam',profile)
                steam2 = self.forward_once(ir,'steam',profile)
                steam = torch.cat([steam1,steam2],1)
                # sio.savemat('features/output.mat', mdict={'data':steam.cpu().numpy()})
            if input_mode == 'RGB+IR':
                steam = torch.cat([x,ir[:,0:1,:,:]],1)
            if input_mode == 'RGB':
                steam = x
            if input_mode == 'IR':
                steam = ir#steam = ir[:,0:1,:,:]
            if input_mode == 'RGB+IR+MF':
                steam = [x,ir[:,0:1,:,:]] #[:,0:1,:,:]
                
            
            self.training |= self.export
            if self.training==True:
                if self.sr:
                    y,output_sr,features = self.forward_once(steam,'yolo', profile) #zjq
                    return y,output_sr,features
                else:
                    y,features = self.forward_once(steam,'yolo', profile) #zjq
                    return y,features
            else:
                y,features = self.forward_once(steam,'yolo', profile) #zjq
                return y[0],y[1],features



    
    def forward_once(self, x, string, profile=False):
        y, dt = [], []  # outputs
        if string == 'steam':
            for m in self.steam:
                if m.f != -1:  # if not from previous layer
                    x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers

                if profile:
                    o = thop.profile(m, inputs=(x,), verbose=False)[0] / 1E9 * 2 if thop else 0  # FLOPS
                    t = time_synchronized()
                    for _ in range(10):
                        _ = m(x)
                    dt.append((time_synchronized() - t) * 100)
                    print('%10.1f%10.0f%10.1fms %-40s' % (o, m.np, dt[-1], m.type))

                x = m(x)  # run
                #y.append(x if m.i in self.save_steam else None)  # save output
            return x
        elif string == 'yolo': 
            # for samDet
            #TO DO SAM should be modified so it output image embedding as well
            # breakpoint()
            # feature_maps = self.sam.image_encoder(x)
            # breakpoint()
            # point_grids = build_all_layer_point_grids(64, 0, 1)
            breakpoint()
            _outputs, feature_maps = self.mask_generator.generate(x)
            breakpoint()
            max_num_boxes = 400
            if len(_outputs) > 400:
                sorted_list = sorted(_outputs, key=lambda x: x['predicted_iou'], reverse=True)
                filtered_boxes = sorted_list[:max_num_boxes]
                _outputs = [d['bbox'] for d in filtered_boxes]
            elif len(_outputs) < 400:
                _outputs = [d['bbox'] for d in _outputs]
                _outputs = _outputs + [[0, 0, 0, 0]] * (max_num_boxes - len(_outputs))
            else:
                _outputs = [d['bbox'] for d in _outputs]
            #TO DO: filter the outputs and take the boxes of 300 or 400
            breakpoint()
            proposals = torch.tensor(_outputs) / 8.0   #8.0 in case image size of 512, 16.0 in case image size of 1024
            rois = deepcopy(proposals)
            rois[:, 2] = rois[:, 0] + rois[:, 2]
            rois[:, 3] = rois[:, 1] + rois[:, 3]
            #rois = proposals
            zero_tensor = torch.zeros((400, 1))
            rois = torch.cat((zero_tensor, rois), dim=1)
            rois = torch.round(rois).to(feature_maps.device)
            breakpoint()
            output_size = (12, 12)

            pooled_features = ops.roi_pool(feature_maps, rois, output_size)
            breakpoint()
            y = self.detect(pooled_features)
            breakpoint()
            # for feature in y[:-1]:
            #     print((torch.numel(feature)-torch.count_nonzero(feature))/torch.numel(feature))

            
            


            self.training |= self.export
            if self.training==True:
                if self.sr:
                    output_sr = self.model_up(y[self.l1],y[self.l2]) #在超分上加attention    
                    return x,output_sr,y#(y[self.f1],y[self.f2],y[self.f3])#(y[4],y[8],y[18],y[21],y[24])#(y[7],y[15],y[-2])
                else:
                    return rois,y#(y[self.f1],y[self.f2],y[self.f3])#(y[4],y[8],y[18],y[21],y[24])#(y[7],y[15],y[-2])
            else:
                return rois,y#(y[17],y[20],y[23])#(y[4],y[8],y[18],y[21],y[24])#(y[7],y[15],y[-2])(y[-4],y[-3],y[-2])


    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.detect     #[-1]  # Detect() module   #*changed self.model[-1] to self.detect

        b_reg = m.bbox_pred.bias.view(1, -1)
        b_reg.data[:, :4] += math.log(0.1)  #adjusting the foreground/background prior

        b_cls = m.cls_score.bias.view(1, -1)  #1 is because we have 1 anchor and not 3 anchors
        b_cls.data += math.log((1 - 0.01) / 0.01)

        m.bboc_pred.bias = torch.nn.Parameter(b_reg.view(-1), requires_grad=True)
        m.cls_score.bias = torch.nn.Parameter(b_cls.view(-1), requires_grad=True)


    def _print_biases(self):
        m = self.model[-1]  # Detect() module
        for mi in m.m:  # from
            b = mi.bias.detach().view(m.na, -1).T  # conv.bias(255) to (3,85)
            print(('%6g Conv2d.bias:' + '%10.3g' * 6) % (mi.weight.shape[1], *b[:5].mean(1).tolist(), b[5:].mean()))

    # def _print_weights(self):
    #     for m in self.model.modules():
    #         if type(m) is Bottleneck:
    #             print('%10.3g' % (m.w.detach().sigmoid() * 2))  # shortcut weights

    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        print('Fusing layers... ')
        for m in self.model.modules():
            if (type(m) is Conv) and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, 'bn')  # remove batchnorm
                m.forward = m.fuseforward  # update forward
        self.info()
        return self

    def nms(self, mode=True):  # add or remove NMS module
        present = type(self.model[-1]) is NMS  # last layer is NMS
        if mode and not present:
            print('Adding NMS... ')
            m = NMS()  # module
            m.f = -1  # from
            m.i = self.model[-1].i + 1  # index
            self.model.add_module(name='%s' % m.i, module=m)  # add
            self.eval()
        elif not mode and present:
            print('Removing NMS... ')
            self.model = self.model[:-1]  # remove
        return self

    def autoshape(self):  # add autoShape module
        print('Adding autoShape... ')
        m = autoShape(self)  # wrap model
        copy_attr(m, self, include=('yaml', 'nc', 'hyp', 'names', 'stride'), exclude=())  # copy attributes
        return m

    def info(self, verbose=False, img_size=640):  # print model information
        model_info(self, verbose, img_size)

def parse_model(d, string, ch,config):  # model_dict, input_channels(3)
    logger.info('\n%3s%18s%3s%10s  %-40s%-30s' % ('', 'from', 'n', 'params', 'module', 'arguments'))
    anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple']
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)

    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    i_shoud_add = 0

    stri = string.split('+')
    if len(stri)==2:
        string_1 = stri[0]
        string_2 = stri[1]
        d_ = d[string_1] + d[string_2]
        save.extend([2,4,5,6,8,9]) #save some layer of backbone
    else:
        d_ = d[stri[-1]]
    if string == 'head':
        ch.append(256)
    for i, (f, n, m, args) in enumerate(d_):  # from, number, module, args
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except:
                pass

        n = max(round(n * gd), 1) if n > 1 else n  # depth gain
        if m in [Conv, ACmix, Bottleneck, SPP, DWConv, MixConv2d, Focus, CrossConv, BottleneckCSP, BottleneckCSP2, SPPCSP, C3, AttentionModel]:
            c1, c2 = ch[f], args[0]

            # Normal
            # if i > 0 and args[0] != no:  # channel expansion factor
            #     ex = 1.75  # exponential (default 2.0)
            #     e = math.log(c2 / ch[1]) / math.log(2)
            #     c2 = int(ch[1] * ex ** e)
            # if m != Focus:

            c2 = make_divisible(c2 * gw, 8) if c2 != no else c2

            # Experimental
            # if i > 0 and args[0] != no:  # channel expansion factor
            #     ex = 1 + gw  # exponential (default 2.0)
            #     ch1 = 32  # ch[1]
            #     e = math.log(c2 / ch1) / math.log(2)  # level 1-n
            #     c2 = int(ch1 * ex ** e)
            # if m != Focus:
            #     c2 = make_divisible(c2, 8) if c2 != no else c2

            args = [c1, c2, *args[1:]]
            if m in [BottleneckCSP, C3,BottleneckCSP2, SPPCSP]:
                args.insert(2, n)
                n = 1
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:# or m is SAM:
            c2 = sum([ch[x if x < 0 else x + 1] for x in f])
        elif m is Detect:
            args.append([ch[x + 1] for x in f])   #*changed removed + 1
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
            print(*args)
            
        elif m is Contract:
            c2 = ch[f if f < 0 else f + 1] * args[0] ** 2
        elif m is Expand:
            c2 = ch[f if f < 0 else f + 1] // args[0] ** 2
        else:
            c2 = ch[f if f < 0 else f + 1]

        if string == 'backbone':
            m_ = m(img_size = args[0], patch_size=args[1], in_chans= args[2], out_chans=args[3], window_size= args[4], depth= 24, num_heads= 16, embed_dim= 1024,global_attn_indexes= [5, 11, 17, 23],)
        else:
            m_ = nn.Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        np = sum([x.numel() for x in m_.parameters()])  # number params
        m_.i, m_.f, m_.type, m_.np = i+i_shoud_add, f, t, np  # attach index, 'from' index, type, number params
        logger.info('%3s%18s%3s%10.0f  %-40s%-30s' % (i+i_shoud_add, f, n, np, t, args))  # print
        save.extend(x % (i+i_shoud_add+0.00001) for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        ch.append(c2)
    if string == 'backbone':
        return layers[0], sorted(save) #*changed  nn.Sequential(*layers) to layers[0]
    return nn.Sequential(*layers), sorted(save)

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     #parser = argparse.ArgumentParser('Set transformer detector', add_help=False)    #newly added
#     parser.add_argument('--temp', default = 'something', type = str, help='it is nothing')
#     parser.add_argument('--cfg', default='yolov5s.yaml', type=str, help='model.yaml')
#     parser.add_argument('--device', default='cuda', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
#     #opt = parser.parse_args()
#     opt, _ = parser.parse_known_args()
#     opt.cfg = check_file(opt.cfg)  # check file
#     set_logging()
#     device = select_device(opt.device)

#     # Create model
#     model = Model(opt.cfg).to(device)
#     model.train()

    # Profile
    # img = torch.rand(8 if torch.cuda.is_available() else 1, 3, 640, 640).to(device)
    # y = model(img, profile=True)

    # Tensorboard
    # from torch.utils.tensorboard import SummaryWriter
    # tb_writer = SummaryWriter()
    # print("Run 'tensorboard --logdir=models/runs' to view tensorboard at http://localhost:6006/")
    # tb_writer.add_graph(model.model, img)  # add model to tensorboard
    # tb_writer.add_image('test', img[0], dataformats='CWH')  # add model to tensorboard
