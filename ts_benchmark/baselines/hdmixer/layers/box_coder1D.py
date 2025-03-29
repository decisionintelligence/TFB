# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import math
import torch
import torch.nn as nn
from torch.nn import functional as F

__all__ = ["pointCoder", "pointwhCoder"]


class pointCoder(nn.Module):
    def __init__(self, input_size, patch_count, weights=(1., 1.,1.), tanh=True):
        super().__init__()
        self.input_size = input_size
        self.patch_count = patch_count
        self.weights = weights
        #self._generate_anchor()
        self.tanh = tanh

    def _generate_anchor(self, device="cpu"):
        anchors = []
        patch_stride_x = 2. / self.patch_count
        for i in range(self.patch_count):
                x = -1+(0.5+i)*patch_stride_x
                anchors.append([x])
        anchors = torch.as_tensor(anchors)
        self.anchor = torch.as_tensor(anchors, device=device)
        #self.register_buffer("anchor", anchors)

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, pts, model_offset=None):
        assert model_offset is None
        self.boxes = self.decode(pts)
        return self.boxes

    def decode(self, rel_codes):
        # print ('xyxy decoding')
        boxes = self.anchor
        pixel = 1./self.patch_count
        wx, wy = self.weights

        dx = F.tanh(rel_codes[:, :, 0]/wx) * pixel if self.tanh else rel_codes[:, :, 0]*pixel / wx
        dy = F.tanh(rel_codes[:, :, 1]/wy) * pixel if self.tanh else rel_codes[:, :, 1]*pixel / wy

        pred_boxes = torch.zeros_like(rel_codes)

        ref_x = boxes[:,0].unsqueeze(0)
        ref_y = boxes[:,1].unsqueeze(0)

        pred_boxes[:, :, 0] = dx + ref_x
        pred_boxes[:, :, 1] = dy + ref_y
        pred_boxes = pred_boxes.clamp_(min=-1., max=1.)

        return pred_boxes

    def get_offsets(self):
        return (self.boxes - self.anchor) * self.input_size


class pointwhCoder(pointCoder):
    def __init__(self, input_size, patch_count, weights=(1., 1.,1.), pts=1, tanh=True, wh_bias=None,deform_range=0.25):
        super().__init__(input_size=input_size, patch_count=patch_count, weights=weights, tanh=tanh)
        self.patch_pixel = pts
        self.wh_bias = None
        if wh_bias is not None:
            self.wh_bias = nn.Parameter(torch.zeros(2) + wh_bias)
        self.deform_range = deform_range
    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, boxes):
        self._generate_anchor(device=boxes.device)
        # print(boxes.shape)
        # print(self.wh_bias.shape)
        if self.wh_bias is not None:
            boxes[:, :, 1:] = boxes[:, :, 1:] + self.wh_bias
        self.boxes = self.decode(boxes)
        points = self.meshgrid(self.boxes)
        return points

    def decode(self, rel_codes):
        # print ('xyxy decoding')
        boxes = self.anchor
        pixel_x = 2./self.patch_count # patch_count=in_size//stride 这里应该用2除而不是1除 得到pixel_x是两个patch中点的原本距离
        wx,  ww1,ww2 = self.weights

        dx = F.tanh(rel_codes[:, :, 0]/wx) * pixel_x/4 if self.tanh else rel_codes[:, :, 0]*pixel_x / wx #中心点不会偏移超过patch_len

        dw1 = F.relu(F.tanh(rel_codes[:, :, 1]/ww1)) * pixel_x*self.deform_range + pixel_x # 中心点左边长度在[stride,stride+1/4*stride]，右边同理
        dw2 = F.relu(F.tanh(rel_codes[:, :, 2]/ww2)) * pixel_x*self.deform_range + pixel_x #
        # dw = 

        pred_boxes = torch.zeros((rel_codes.shape[0],rel_codes.shape[1],rel_codes.shape[2]-1)).to(rel_codes.device)

        ref_x = boxes[:,0].unsqueeze(0)

        pred_boxes[:, :, 0] = dx + ref_x - dw1
        pred_boxes[:, :, 1] = dx + ref_x + dw2
        pred_boxes = pred_boxes.clamp_(min=-1., max=1.)

        return pred_boxes


    
    def meshgrid(self, boxes):
        B = boxes.shape[0]
        xs= boxes
        xs = torch.nn.functional.interpolate(xs, size=self.patch_pixel, mode='linear', align_corners=True)
        results = xs
        results = results.reshape(B, self.patch_count,self.patch_pixel, 1)
        #print((1+results[0])/2*336)
        return results
