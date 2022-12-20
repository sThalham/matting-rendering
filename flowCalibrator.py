import sys, os, argparse
import numpy as np
#from scipy.misc import imread, imsave
import render_utils as utils
import cv2

class FlowCalibrator:
    def __init__(self, imgs, out_dir, out_name):
        self.imgs = imgs
        self.out_dir = out_dir
        self.out_name = out_name
        self.h = imgs[0].shape[0]
        self.w = imgs[0].shape[1]
        self.mask = imgs[0]
        self.rho  = imgs[1]

    def obtainImgBinaryCode(self, sub_imgs, h, w):
        binary_code = np.chararray((h, w)); binary_code[:]=''
        for img in sub_imgs:
            bit_code = np.chararray((h,w), itemsize=1)
            bit_code[img >  190] = '1'
            bit_code[img <= 190] = '0'
            binary_code = binary_code + bit_code
        return binary_code

    def findCorrespondence(self):
        self.flow_x_idx = np.zeros((self.h, self.w), dtype=np.int64)
        self.flow_y_idx = np.zeros((self.h, self.w), dtype=np.int64)
        self.x_grid = np.tile(np.linspace(0, self.w-1, self.w), (self.h, 1)).astype(int)
        self.y_grid = np.tile(np.linspace(0, self.h-1, self.h), (self.w, 1)).T.astype(int)
        self.img_code  = self.obtainImgBinaryCode(self.imgs[2:], self.h, self.w)
        self.findCorrespondenceGraycode()

    def findCorrespondenceGraycode(self):
        digit_code = [int(code, 2) for code in self.img_code.flatten()]
        digit_code = np.array(digit_code).reshape(self.h, self.w)
        self.flow_x_idx = np.mod(digit_code, self.w)
        self.flow_y_idx = np.divide(digit_code, self.w)

        self.flow_x_idx -= self.x_grid
        self.flow_y_idx -= self.y_grid
        self.flow_x_idx[self.mask >= 200] = 0
        self.flow_y_idx[self.mask >= 200] = 0
        #self.flow_x_idx[self.mask >= 200] = 255
        #self.flow_y_idx[self.mask >= 200] = 255
        self.saveFlow(self.flow_x_idx, self.flow_y_idx, self.rho)

    def flowWithRho(self, flow_color, rho):
        h = rho.shape[0]
        w = rho.shape[1]
        flow_rho = flow_color * np.tile(rho.reshape(h,w,1), (1,1,3)).astype(float) /255
        return flow_rho.astype(np.uint8)

    def writeFlowBinary(self, flow, filename):
        flow = flow.astype(np.float32)
        with open(filename, 'wb') as f:
            magic = np.array([202021.25], dtype=np.float32)
            h_w   = np.array([flow.shape[0], flow.shape[1]], dtype=np.int32)
            magic.tofile(f)
            h_w.tofile(f)
            flow.tofile(f)

    def saveFlow(self, flow_x, flow_y, rho):
        h = flow_x.shape[0]
        w = flow_x.shape[1]
        flow = np.zeros((h,w,2))
        flow[:,:,1] = flow_x; flow[:,:,0] = flow_y
        flow_color = utils.flowToColor(flow)
        cv2.imwrite(os.path.join(self.out_dir, self.out_name + '.png'), self.flowWithRho(flow_color, rho))
        utils.writeFlowBinary(flow, os.path.join(self.out_dir, self.out_name + '.flo'))

