# Loss functions

import torch
import torch.nn as nn


import math
import torch.nn.functional as F
from .general import xyxy2xywh, xywh2xyxy,xywhn2xyxy
from .general import bbox_iou
from .torch_utils import is_parallel
def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


class BCEBlurWithLogitsLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self, alpha=0.05):
        super(BCEBlurWithLogitsLoss, self).__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none')  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)  # prob from logits
        dx = pred - true  # reduce only missing label effects
        # dx = (pred - true).abs()  # reduce missing label and false label effects
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class QFocalLoss(nn.Module):
    # Wraps Quality focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(QFocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)

        pred_prob = torch.sigmoid(pred)  # prob from logits
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class ComputeLoss:
    # Compute losses
    def __init__(self, model, autobalance=False):
        super(ComputeLoss, self).__init__()
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=0.0)

        # Focal loss
        g = h['fl_gamma']  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        det = model.module.detect if is_parallel(model) else model.detect  # Detect() module        #*changed model.model[-1]   to model.detect
        self.balance = {3: [4.0, 1.0, 0.4]}.get(1, [4.0, 1.0, 0.25, 0.06, .02])  # P3-P7  #*changed det.nl changed to 1
        self.ssi = 0    #list(det.stride).index(16) if autobalance else 0  # stride 16 index      #*changed hardcoded to 0
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, model.gr, h, autobalance
        self.mse = nn.MSELoss()
        #TODO  remove below for loop and hard code
        self.na = None
        self.nc = det.nc
        self.nl = 1
        self.anchors = None
        # for k in 'na', 'nc', 'nl', 'anchors':
        #     breakpoint()
        #     setattr(self, k, getattr(det, k))

    def __call__(self, p, propos, targets):  # predictions, targets, model
        device = targets.device
        lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
        tcls, tbox, tcord, indices, proposals, n_indices, thr_match = self.build_targets(propos, targets, p.shape[1])  # targets

        # Losses
        
        for i, pi in enumerate(p):  # layer index, layer predictions
            b = indices[i]  # #, a    image, proposal   --image, anchor, gridy, gridx

            obj = n_indices == 1  #    [..., 0]    in paper this is Iobj_i
            noobj = n_indices == 0  #   [..., 0]    in paper this is Inoobj_i


            tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                ps = pi[b]  # prediction subset corresponding to targets

                # Regression

                #pxy = ps[:, :2] * 2. + proposals[b, a, 1:3] #.sigmoid()   #*changed * 2. - 0.5 #取出每幅图像包含目标的网格和对应anchor的预测结果，对目标位置的预测值进行sigmoid运算后乘以2再减去0.5得到box中心点的位置
                predicted_boxes = torch.zeros_like(proposals[b, 0:])
                #ps[:, :4] = ps[:, :4].sigmoid()
                predicted_boxes[:, 0] = ps[:, 0].sigmoid() * (proposals[b, 2] + 1e-6) + proposals[b, 0]
                predicted_boxes[:, 1] = ps[:, 1].sigmoid() * (proposals[b, 3]+ 1e-6) + proposals[b, 1]
                pxy = ps[:, :2].sigmoid()
                pwh = ps[:, 2:4]    #.sigmoid()   + proposals[b, a, 3:5]  #.sigmoid()     #*changed (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i] #对目标宽高预测值进行sigmoid运算后乘以2再平方再乘以对应的anchor的宽高得到预测框的宽高值
                predicted_boxes[:, 2] = torch.exp(ps[:, 2]) * (proposals[b, 2] + 1e-6)
                predicted_boxes[:, 3] = torch.exp(ps[:, 3]) * (proposals[b, 3] + 1e-6)
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                iou = bbox_iou(predicted_boxes.T, tbox[i], x1y1x2y2=False, CIoU=True)  # iou(prediction, target) .T 矩阵转置
                
                lbox += (1.0 - iou).mean()  # self.mse(pbox, tcord[i])    # iou loss   F.mse_loss(pbox, tbox[i])#

                # Objectness

                tobj[b] = (1.0 - self.gr) + self.gr * iou.detach().clamp(0).type(tobj.dtype)  # iou ratio
                
                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    # t = torch.full_like(ps[:, 5:], self.cn, device=device)  # targets
                    
                    # t[range(n), tcls[i]] = self.cp
                    # lcls += self.BCEcls(ps[:, 5:], t)  # BCE
                    
                    #new cls loss here
                    t = torch.full_like(pi[:, 4:], self.cn, device=device)
                    t[b, tcls[i]] = 1
                    t = t[:, :self.nc]

                    lcls += self.BCEcls(pi[:, 5:], t)
                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

            #obji = self.BCEobj(pi[b][..., 4], tobj[b])     #ps
            #obji += self.BCEobj(pi[..., 4][noobj], tobj[noobj])    #nb, na
            lobj += 0    #obji * self.balance[i]  # obj loss

            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls = lcls / p.shape[1]
        lcls *= self.hyp['cls']
        bs = 8#tobj.shape[0]  # batch size

        loss = lbox + lobj + lcls
        return loss * bs, lbox , lobj , lcls, thr_match #loss * bs, torch.cat((lbox, lobj, lcls, loss)).detach() # .detach_() 和 .data用于切断反向传播

    def build_targets(self, p, targets, batch_size):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        '''
        changed from original build_targets
        inputs:
            -p: prposals not predictiosn, in shape of b x n x 4 or n x 5 (an additional showing batch number)
            -targets (image, class, x,y,w,h)
            -batch_size: batch size
        outputs:
            -indices: list of tuples including tensors one for batch index / image id, and another for
            proposal index. used to get proposal specific for subsequent tbox and tcls
            -tbox: list of tensors including difference xywh of each ground truth from corresponding
            proposal
            -tcls: list of tensors including class values corresponding to relevant tbox
            -tcord: list of tensors including the curated xywh coordinates for positive proposals
            -indices: list of tensor including the index number of positive proposal
            -p: proposals in n X 4 format
            -objectness: is an experimental list of tensor including all proposal indexes inlcuding 1 for positive 0 for negative and -1 for ignored proposals
            -thr_match: number of proposals that matched the targets based on iou threshold
        '''
        #p should be the output of SAM; we match it with the targets having shape of [batch, x, y, w, h]
        #assum p is output of SAM
        thr_match = 0
        for i, pro in enumerate(p):
            pro[:, 0:4] = xyxy2xywh(pro[:, 0:4])
            pro[:, 0:4] = pro[:, 0:4] /64.0
            p[i] = pro
        # p = p.view(-1, 5)  #below xyxy2xywh requires n x 4 shape not n x n x 4
        # p[:, 1:5] = xyxy2xywh(p[:, 1:5])
        # p = p.view(-1, 400, 5)

        # p[:,:, 1:5] = p[:, :, 1:5] / 64.0

        
        #raising targets from 0-1 to 64  according to proposals and feature maps
        #targets[:, 2: 6] = targets[:, 2:6] * 64
        objectness = torch.zeros(batch_size)
       
        tars = torch.zeros(targets.shape[0]).to(targets.device)  #track keepr of which target has proposal or not
        tcls, tbox, indices, anch = [], [], [], []
        n_indices =[]

        tbox = []
        tcord = []
        tcls = []  #8 is the number of classes
        # list including all positive proposals index
        batches =[]
        proposal_idx = []
        n_batches = []
        n_proposalIdx = []
        #Integer tracker to track up to which index in proposals have been processed to create tensors of n x 1 for batches
        indx_tracker = 0 
        #list to track which batche's targets count has been added to indx_tracker
        batch_list = []
        if targets.shape[0] >0 and int(targets[0, 0]) != 0:
            for i in range(int(targets[0, 0])):
                indx_tracker += p[i].shape[0]
                batch_list.append(i)
        for i, target in enumerate(targets):
            if len(batch_list) and int(target[0]) > batch_list[-1] +1 :
                for j in range( batch_list[-1] +1, int(target[0])):
                    indx_tracker += p[j].shape[0]
                    batch_list.append(j)
            preds = torch.zeros(p[int(target[0])].shape[0]).to(targets.device)   #keep track of which proposal is assigned or not
            iou_p = bbox_iou(target[2:6], p[int(target[0])][:,0:4], x1y1x2y2=False, CIoU=False)  #iou between target and proposals of same image [target[0]] as index for proposals of same image
            iou_indices = iou_p.argsort(descending=True, dim=0)
            for iou_index in iou_indices:
                p_taken = preds[iou_index] == 1
                tar_taken = tars[i] == 1
                if not p_taken and not tar_taken:
                    
                    if iou_p[iou_index] > 0.5:
                        thr_match = thr_match + 1
                    #tbox[iou_index][0] = 1  #objectness
                        preds[iou_index] = 1    #proposal taken
                        tars[i] = 1             #target has a proposal
                        #attaching ground truth xywh to proposals
                        xy = [target[2] , target[3]]  #  - p[int(target[0]), iou_index, 1]   - p[int(target[0]), iou_index, 2]  - p[1]difference of ground-truth and proposal xy
                        wh = [target[4], target[5]] #   - p[int(target[0]), iou_index, 3] - p[int(target[0]), iou_index, 4] - p[2] - p[3] difference of ground-truth and porposal wh
                        x_coord = [(target[2] - p[int(target[0])][iou_index, 0])/p[int(target[0])][iou_index, 2]]
                        y_coord = [(target[3] - p[int(target[0])][iou_index, 1])/p[int(target[0])][iou_index, 3]]
                        w_coord = [torch.log(target[4] /p[int(target[0])][iou_index, 2])]
                        h_coord = [torch.log(target[5] /p[int(target[0])][iou_index, 3])]
                        # if indx_tracker + iou_index>batch_size:
                        #     breakpoint()
                        #cls[target[1]] = 1   #corresponding class index set to 1 others zero
                        batches.append(indx_tracker + iou_index)   #batch index / image id

                        objectness[indx_tracker + iou_index] = 1

                        box = torch.cat((torch.tensor(xy), torch.tensor(wh)))
                        box_coord = torch.cat((torch.tensor(x_coord), torch.tensor(y_coord), torch.tensor(w_coord), torch.tensor(h_coord)))
                        tbox.append(box)
                        tcord.append(box_coord)
                        tcls.append(target[1].long())
                elif preds[iou_index] == 0 and iou_p[iou_index] >0.4:
                     n_batches.append(indx_tracker + iou_index)
                     objectness[indx_tracker + iou_index] = -1
            #update indx_tracker

            if not int(target[0]) in batch_list and i + 1 <targets.shape[0] and targets[i+1][0]> target[0]:
                
                indx_tracker += p[int(target[0])].shape[0]

                batch_list.append(int(target[0]))

        # for i in range(targets.shape[0]):
        #     n = batches.count(i) * 2
        #     tar_indices = [proposal_idx[j] for j in range(len(batches)) if batches[j] == i]
        #     rand = torch.randperm(100)[:n]
        #     rand = rand[~torch.isin(rand, torch.tensor(tar_indices))]
        #     n1 = n // 3
        #     n_batches.extend([i] * rand.shape[0])
        #     n_proposalIdx.extend(rand.tolist())
        #     if n1:
        #         rand1 = torch.randperm(299)[:n1] + 100
        #         rand1 = rand1[~torch.isin(rand1, torch.tensor(tar_indices))]
        #         n_batches.extend([i] * rand1.shape[0])
        #         n_proposalIdx.extend(rand1.tolist())
        if tbox:
            tbox = torch.stack(tbox).to(targets.device)
            tbox = [tbox]
            tcord = torch.stack(tcord).to(targets.device)
            tcord = [tcord]
        if tcls:
            tcls = torch.stack(tcls).to(targets.device)
            tcls = [tcls]
        indices.append((torch.tensor(batches).long().to(targets.device)))   #, torch.tensor(proposal_idx).to(targets.device)
        #n_indices.append((torch.tensor(n_batches).long().to(targets.device), torch.tensor(n_proposalIdx).to(targets.device)))
        p = torch.cat(p).to(targets.device)
        return tcls, tbox, tcord, indices, p, objectness, thr_match



class LevelAttention_loss(nn.Module):

    def forward(self, img_batch_shape, attention_mask, target):

        h, w = img_batch_shape[2], img_batch_shape[3]

        #mask_losses = []
        mask_loss = 0

        batch_size = img_batch_shape[0]
        n = target.shape[0]  # number of targets
        if n:
            for j in range(batch_size):
                try:
                    n = min([i for i in range(target.shape[0]) if target[i,0]==j])
                    m = max([i for i in range(target.shape[0]) if target[i,0]==j])
                except:
                    continue
                
                bbox_annotation = xywhn2xyxy(target[n:m+1,2:], w=w, h=h, padw=0, padh=0) #bboxs[j, :, :]
                #bbox_annotation = bbox_annotation[bbox_annotation[:, 4] != -1]

                cond1 = torch.le(bbox_annotation[:, 0], w)
                cond2 = torch.le(bbox_annotation[:, 1], h)
                cond3 = torch.le(bbox_annotation[:, 2], w)
                cond4 = torch.le(bbox_annotation[:, 3], h)
                cond = cond1 * cond2 * cond3 * cond4

                bbox_annotation = bbox_annotation[cond, :]

                # if bbox_annotation.shape[0] == 0:
                #     mask_losses.append(torch.tensor(0).float().cuda())
                #     continue

                #bbox_area = (bbox_annotation[:, 2] - bbox_annotation[:, 0]) * (bbox_annotation[:, 3] - bbox_annotation[:, 1])

                # mask_loss = []

                #for id in range(len(attention_mask)):
                ################
                #attention_map = attention_mask[id][j, 0, :, :]
                attention_map = attention_mask[j, 0, :, :]

                # min_area = (2 ** (id + 5)) ** 2 * 0.5
                # max_area = (2 ** (id + 5) * 1.58) ** 2 * 2

                # level_bbox_indice1 = torch.ge(bbox_area, min_area)
                # level_bbox_indice2 = torch.le(bbox_area, max_area)

                # level_bbox_indice = level_bbox_indice1 * level_bbox_indice2

                # level_bbox_annotation = bbox_annotation[level_bbox_indice, :].clone()

                level_bbox_annotation = bbox_annotation.clone()

                attention_h, attention_w = attention_map.shape

                if level_bbox_annotation.shape[0]:
                    level_bbox_annotation[:, 0] *= attention_w / w
                    level_bbox_annotation[:, 1] *= attention_h / h
                    level_bbox_annotation[:, 2] *= attention_w / w
                    level_bbox_annotation[:, 3] *= attention_h / h

                mask_gt = torch.zeros(attention_map.shape)
                mask_gt = mask_gt.cuda()

                for i in range(level_bbox_annotation.shape[0]):

                    x1 = max(int(level_bbox_annotation[i, 0]), 0)
                    y1 = max(int(level_bbox_annotation[i, 1]), 0)
                    x2 = min(math.ceil(level_bbox_annotation[i, 2]) + 1, attention_w)
                    y2 = min(math.ceil(level_bbox_annotation[i, 3]) + 1, attention_h)

                    mask_gt[y1:y2, x1:x2] = 1

                mask_gt = mask_gt[mask_gt >= 0]
                mask_predict = attention_map[attention_map >= 0]

                #mask_loss.append(F.binary_cross_entropy(mask_predict, mask_gt))
                #mask_loss.append(nn.BCEWithLogitsLoss()(mask_predict, mask_gt))
                mask_loss += nn.BCEWithLogitsLoss()(mask_predict, mask_gt)
                #################
                #mask_losses.append(torch.stack(mask_loss).mean())

        return mask_loss #torch.stack(mask_losses).mean(dim=0, keepdim=True)
