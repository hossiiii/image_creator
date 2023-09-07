import PySimpleGUI as sg
import requests
import uuid
import hashlib

sg.theme('LightGray6')
sg.popup_quick_message('\n\n画像クリエイターを起動中です...\n\n初回起動時は時間がかかることがあります\n\n',image = "./asset/gui/setup.png")

#環境構築
import os
import pip, site, importlib
pip.main(['install', '-U', 'pip'])
importlib.reload(site)
pip.main(['install','GitPython'])
pip.main(['install','wget'])
pip.main(['install','configparser'])
importlib.reload(site)
import git
import wget
import configparser

##ディレクトリ変数定義
home_path = os.getcwd()
yolox_path = home_path + '/YOLOX'
rename_image_path = home_path + '/image/rename_image'
os.makedirs(rename_image_path, exist_ok=True)
resize_image_path = home_path + '/image/resize_image'
os.makedirs(resize_image_path, exist_ok=True)
display_image_path = home_path + '/image/display_image'
os.makedirs(display_image_path, exist_ok=True)
sorce_image_path = home_path + '/image/sample_image' #指定可
os.makedirs(sorce_image_path, exist_ok=True)
result_image_path = home_path + '/image/result_image' #指定可
os.makedirs(result_image_path, exist_ok=True)

##設定変数定義
config = configparser.ConfigParser()
config.read('./asset/conf/conf.ini')
pixcel = int(config['SECTION1']['pixcel'])
output_max_num = int(config['SECTION1']['output_max_num'])
person_frame_percent = float(config['SECTION1']['person_frame_percent'])
product_frame_percent = float(config['SECTION1']['product_frame_percent'])
frame_file_name = config['SECTION1']['frame_file_name']

tag1_file_name = config['SECTION1']['tag1_file_name']
tag1_percent_min = float(config['SECTION1']['tag1_percent_min']) if config['SECTION1']['tag1_percent_min'] != "" else config['SECTION1']['tag1_percent_min']
tag1_percent_max = float(config['SECTION1']['tag1_percent_max']) if config['SECTION1']['tag1_percent_max'] != "" else config['SECTION1']['tag1_percent_max']
if config['SECTION1']['tag1_position'] != "":
  tag1_position = config['SECTION1']['tag1_position'][1:-1]
  tag1_position = [int(i) for i in tag1_position.split(",")]
else:
  tag1_position = config['SECTION1']['tag1_position']
tag2_file_name = config['SECTION1']['tag2_file_name']
tag2_percent_min = float(config['SECTION1']['tag2_percent_min']) if config['SECTION1']['tag2_percent_min'] != "" else config['SECTION1']['tag2_percent_min']
tag2_percent_max = float(config['SECTION1']['tag2_percent_max']) if config['SECTION1']['tag2_percent_max'] != "" else config['SECTION1']['tag2_percent_max']
if config['SECTION1']['tag2_position'] != "":
  tag2_position = config['SECTION1']['tag2_position'][1:-1]
  tag2_position = [int(i) for i in tag2_position.split(",")]
else:
  tag2_position = config['SECTION1']['tag2_position']
tag3_file_name = config['SECTION1']['tag3_file_name']
tag3_percent_min = float(config['SECTION1']['tag3_percent_min']) if config['SECTION1']['tag3_percent_min'] != "" else config['SECTION1']['tag3_percent_min']
tag3_percent_max = float(config['SECTION1']['tag3_percent_max']) if config['SECTION1']['tag3_percent_max'] != "" else config['SECTION1']['tag3_percent_max']
if config['SECTION1']['tag3_position'] != "":
  tag3_position = config['SECTION1']['tag3_position'][1:-1]
  tag3_position = [int(i) for i in tag3_position.split(",")]
else:
  tag3_position = config['SECTION1']['tag3_position']

sorce_image_path = config['SECTION1']['sorce_image_path'] if config['SECTION1']['sorce_image_path'] != "" else sorce_image_path
result_image_path = config['SECTION1']['result_image_path'] if config['SECTION1']['result_image_path'] != "" else result_image_path

try:
    git.Repo.clone_from('https://github.com/Megvii-BaseDetection/YOLOX',"YOLOX") #一緒にディレクトリも作成する
except:
    print('YOLOXクローンあり')

os.chdir('YOLOX')
pip.main(['install', '-r', 'requirements.txt'])
pip.main(['install', '-v', '-e', '.'])
pip.main(['install','cython'])
pip.main(['install','git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'])
importlib.reload(site)

if(os.path.isfile("yolox_x.pth")):
    print("yolox_x.pthあり")
else:
    print("yolox_x.pthなし")
    site_url = 'https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_x.pth'
    wget.download(site_url,"yolox_x.pth")

pip.main(['install','opencv-python'])
pip.main(['install','opencv-contrib-python'])
pip.main(['install','pillow'])
pip.main(['install','loguru'])
pip.main(['install','torch'])
pip.main(['install','torchvision'])
pip.main(['install','thop'])
pip.main(['install','tabulate'])
pip.main(['install','tqdm'])

import cv2
import numpy as np
from PIL import Image
import glob
import argparse
import time
from loguru import logger
import torch
import itertools
import math
import glob
import shutil

os.chdir(yolox_path)
from yolox.data.data_augment import ValTransform
from yolox.data.datasets import COCO_CLASSES
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess, vis
os.chdir(home_path)


def match_file(dir_path:str,image_type:str):
  ''' 条件に一致したファイルのリストを作成 '''
  files = os.listdir(dir_path)
  files_file = [f for f in files if os.path.isfile(os.path.join(dir_path, f))]
  files = [s for s in files_file if image_type in s]
  return files

def get_parameters(filename:str):
  ''' ファイル名から、int形式でw/h/pxを返す '''
  w = filename[filename.find("w")+1:filename.find("h")]
  h = filename[filename.find("h")+1:filename.find("px")]
  px = filename[filename.find("px")+2:filename.find(".")]
  id = filename[filename.rfind("-")+1:filename.find("_")]
  return [int(w),int(h),int(px),str(id)]

def delete_files(dir_list:list):
  for dir in dir_list:
    shutil.rmtree(dir)
    os.mkdir(dir)

def pil2cv(image):
    ''' PIL型 -> OpenCV型 '''
    new_image = np.array(image, dtype=np.uint8)
    if new_image.ndim == 2:  # モノクロ
        pass
    elif new_image.shape[2] == 3:  # カラー
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
    elif new_image.shape[2] == 4:  # 透過
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGBA2BGRA)
    return new_image

def cv2pil(image):
    ''' OpenCV型 -> PIL型 '''
    new_image = image.copy()
    if new_image.ndim == 2:  # モノクロ
        pass
    elif new_image.shape[2] == 3:  # カラー
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
    elif new_image.shape[2] == 4:  # 透過
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGRA2RGBA)
    new_image = Image.fromarray(new_image)
    return new_image

def to_RGB(image:Image):
  ''' RGBα -> RGB '''
  if image.mode == 'RGB': return image
  background = Image.new("RGB", image.size, (255, 255, 255))
  background.paste(image, mask=image.split()[3])  # 3 is the alpha channel
  background.format = image.format
  return background

def find_edge(image:Image,image_type:str):
  ''' エッジ検出 '''
  new_image = np.array(image, dtype=np.uint8)
  img = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
  blur = cv2.blur(img,(5,5))
  if(image_type == 'person'):
    edges = cv2.Canny(blur,150,250)
  else:
    edges = cv2.Canny(blur,50,55)

  return edges

def find_target(edges): 
  ''' エッジから短形を算出　'''
  results = np.where(edges==255)
  top = np.min(results[0])
  bottom = np.max(results[0]) - 1
  left = np.min(results[1])
  right = np.max(results[1]) - 1
  return (left,top,right,bottom)

def to_extrapolate(left:int,top:int,right:int,bottom:int,image:Image):
  ''' ４辺がエッジでなければ背景を拡張　'''
  img_right, img_bottom = image.size
  img_right = img_right - 2 #なぜか2pixcelずれるため調整
  img_bottom = img_bottom - 2 #なぜか2pixcelずれるため調整
  img_left = 0
  img_top = 0

  img_width = img_right - img_left
  img_height = img_bottom - img_top

  width = right - left
  height = bottom - top

  arr = [img_left + left, img_top + top, img_right - right, img_bottom - bottom]
  n = sum(1 for c in arr if c<=0)

  if(n == 0):
    w, h = image.size
    cv2_image = pil2cv(image)
    M = np.array([[1, 0, w],
                [0, 1, h]], dtype=float)
    cv2_image = cv2.warpAffine(cv2_image, M, dsize=(w * 3, h * 3),borderMode=cv2.BORDER_REPLICATE)
    pil_image = cv2pil(cv2_image)
    # plt.imshow(pil_image)
    return pil_image
  else:
    return image

def find_target_margin(img_left:int,img_top:int,img_right:int,img_bottom:int,left:int,top:int,right:int,bottom:int,frame_percent:float):
  ''' 余白をつけた短形を算出　'''
  width = right - left
  height = bottom - top

  frame = 0

  if(height > width):
    frame = height * frame_percent
  else:
    frame = width * frame_percent

  arr = [img_left + left - frame, img_top + top - frame, img_right - right - frame, img_bottom - bottom - frame]
  n = sum(1 for c in arr if c<=0)
  arr2 = [img_left + left - frame*2, img_top + top - frame*2, img_right - right - frame*2, img_bottom - bottom - frame*2]
  n2 = sum(1 for c in arr2 if c<=0)

  #余白をつけられない場合（２箇所以上がフレーム含めてはみ出していないか）余白をつけずにリターン
  if (n== 0):
    return (left-frame,top-frame,right+frame,bottom+frame,n) #フレームを追加してリターン
  elif (n== 1):
    if (n2>1): # 枠の２倍した時にはみ出る場合はそのままリターン
      return (left,top,right,bottom,n)
    elif (arr[0]<=0): # left
      return (left,top-frame,right+frame*2,bottom+frame,n)
    elif (arr[1]<=0): # top
      return (left-frame,top,right+frame,bottom+frame*2,n)
    elif (arr[2]<=0): # right
      return (left-frame*2,top-frame,right,bottom+frame,n)
    elif (arr[3]<=0): # bottom
      return (left-frame,top-frame*2,right+frame,bottom,n)
  else:
    return (left,top,right,bottom,n) #２箇所以上はみでるのでフレームを追加せずにリターン

def find_target_aspect(left:int,top:int,right:int,bottom:int,x:int,y:int):
  ''' 指定したアスペクト比をつけた短形を算出　'''
  width = right - left
  height = bottom - top

  if(y > x):
    rate = x / y
    margin = (height * rate - width ) / 2

    if(margin >= 0):
      left = left - margin
      right = right + margin
    else:
      margin2 = (width * 1/rate - height) / 2
      top = top - margin2
      bottom = bottom + margin2

  elif(x > y):
    rate = y / x
    margin = (width * rate - height ) / 2

    if(margin >= 0):
      top = top - margin
      bottom = bottom + margin
    else:
      margin2 = (height * 1/rate - width) / 2
      left = left - margin2
      right = right + margin2

  else:
    rate = 1
    if(height > width):
      margin = (height - width)/2
      if(margin >= 0):
        left = left - margin
        right = right + margin
      else:
        margin2 = (width - height)/2
        top = top - margin2
        bottom = bottom + margin2

    elif(width > height):
      margin = (width - height)/2
      if(margin >= 0):
        top = top - margin
        bottom = bottom + margin
      else:
        margin2 = (height - width)/2
        left = left - margin2
        right = right + margin2

    else:
      margin = (width - height)/2
      if(margin >= 0):
        top = top - margin
        bottom = bottom + margin
      else:
        margin2 = (height - width)/2
        left = left - margin2
        right = right + margin2

  return (left,top,right,bottom,margin)

def to_crop(image:Image,left:int,top:int,right:int,bottom:int):
  ''' pil + 短形 -> cropping pil　'''
  trim_img = image.crop((left, top, right, bottom))
  return trim_img

def print_size(memo:str,left:int,top:int,right:int,bottom:int):
  ''' print size infomation　'''
  width = right - left
  height = bottom - top
  print(memo + " left: " + str(left) + " top: " + str(top) + " right: " + str(right) + " bottom: " + str(bottom))
  print(memo + " width: " + str(width) + " height: " + str(height))

def get_resize_image(img_path:str,x_aspect:int,y_aspect:int,frame_percent:float,image_type:str,out_path:str):
  rgb_img = to_RGB(Image.open(img_path))

  edges = find_edge(rgb_img,image_type)
  left,top,right,bottom = find_target(edges)

  rgb_img = to_extrapolate(left,top,right,bottom,rgb_img) #もし上下左右がくっついてなければ元画像をを拡張しておく

  img_right, img_bottom = rgb_img.size
  img_right = img_right - 2 #なぜか2pixcelずれるため調整
  img_bottom = img_bottom - 2 #なぜか2pixcelずれるため調整
  img_left = 0
  img_top = 0

  edges = find_edge(rgb_img,image_type)
  left,top,right,bottom = find_target(edges)
  left,top,right,bottom,contact_num = find_target_margin(img_left,img_top,img_right,img_bottom,left,top,right,bottom,frame_percent)
  left,top,right,bottom,aspect_margin = find_target_aspect(left,top,right,bottom,x_aspect,y_aspect)

  # if (aspect_margin>=0 and image_type == 'person') or (aspect_margin>=0 and image_type == 'product' and contact_num < 2):
  if (aspect_margin>=0 and image_type == 'person') or (image_type == 'product' and contact_num < 2):
    trim_img = to_crop(rgb_img,left, top, right, bottom)
    trim_img_resize = trim_img.resize((x_aspect,y_aspect))
    out_path = out_path[:out_path.rfind('.')] + '.png'
    trim_img_resize.save(out_path,'png', quality=100)
    return aspect_margin
  else:
    return aspect_margin

#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]

#####修正点#####
def make_parser(sorce_image_path:str):
    parser = argparse.ArgumentParser("YOLOX Demo!")
    parser.add_argument(
        "--demo", default="image", help="demo type, eg. image, video and webcam"
    )
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default="yolox-x", help="model name")
    parser.add_argument(
        "--path", default=sorce_image_path, help="path to images or video"
    )
    parser.add_argument("--camid", type=int, default=0, help="webcam demo camera id")
    parser.add_argument(
        "--save_result",
        action="store_true",
        help="whether to save the inference result of image/video",
    )

    # exp file
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="pls input your experiment description file",
    )
    parser.add_argument("-c", "--ckpt", default="yolox_x.pth", type=str, help="ckpt for eval")

    if(torch.cuda.is_available()):
      parser.add_argument(
          "--device",
          default="gpu",
          type=str,
          help="device to run our model, can either be cpu or gpu",
      )
    else:
      parser.add_argument(
          "--device",
          default="cpu",
          type=str,
          help="device to run our model, can either be cpu or gpu",
      )

    parser.add_argument("--conf", default=0.5, type=float, help="test conf")
    parser.add_argument("--nms", default=0.45, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=160, type=int, help="test img size")
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision evaluating.",
    )
    parser.add_argument(
        "--legacy",
        dest="legacy",
        default=False,
        action="store_true",
        help="To be compatible with older versions",
    )
    parser.add_argument(
        "--fuse",
        dest="fuse",
        default=False,
        action="store_true",
        help="Fuse conv and bn for testing.",
    )
    parser.add_argument(
        "--trt",
        dest="trt",
        default=False,
        action="store_true",
        help="Using TensorRT model for testing.",
    )
    return parser


def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            ext = os.path.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names

class Predictor(object):
    def __init__(
        self,
        model,
        exp,
        cls_names=COCO_CLASSES,
        trt_file=None,
        decoder=None,
        device="cpu",
        fp16=False,
        legacy=False,
    ):
        self.model = model
        self.cls_names = cls_names
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        self.preproc = ValTransform(legacy=legacy)
        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, exp.test_size[0], exp.test_size[1]).cuda()
            self.model(x)
            self.model = model_trt

    def inference(self, img):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = os.path.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        ratio = min(self.test_size[0] / img.shape[0], self.test_size[1] / img.shape[1])
        img_info["ratio"] = ratio

        img, _ = self.preproc(img, None, self.test_size)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.float()
        if self.device == "gpu":
            img = img.cuda()
            if self.fp16:
                img = img.half()  # to FP16

        with torch.no_grad():
            t0 = time.time()
            outputs = self.model(img)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(
                outputs, self.num_classes, self.confthre,
                self.nmsthre, class_agnostic=True
            )
        return outputs, img_info

    def visual(self, output, img_info, cls_conf=0.35):
        ratio = img_info["ratio"]
        img = img_info["raw_img"]
        if output is None:
            return img
        output = output.cpu()

        bboxes = output[:, 0:4]

        # preprocessing: resize
        bboxes /= ratio

        cls = output[:, 6]
        scores = output[:, 4] * output[:, 5]

        vis_res = vis(img, bboxes, scores, cls, cls_conf, self.cls_names)
        return vis_res

#####修正点#####
def image_demo(predictor, vis_folder, path, current_time, save_result, rename_image_path:str):
    if os.path.isdir(path):
        files = get_image_list(path)
    else:
        files = [path]
    files.sort()

    count_person = 0
    count_product = 0

    for image_name in files:
        if(image_name[image_name.rfind('/')+1:] != tag1_file_name and image_name[image_name.rfind('/')+1:] != tag2_file_name and image_name[image_name.rfind('/')+1:] != tag3_file_name and image_name[image_name.rfind('/')+1:] != frame_file_name):
          root, ext = os.path.splitext(image_name)
          outputs, img_info = predictor.inference(image_name)
          result_image = predictor.visual(outputs[0], img_info, predictor.confthre)
          save_file_name = rename_image_path + "/"
          if(outputs[0] != None): #何か検知したら
              # print(outputs[0])
              c_person = 0
              c_product = 0

              for cat in outputs[0].tolist():
                  if(cat[6] == 0):
                      # print("画像に人あり")
                      c_person += 1
                  else:
                      # print("画像に商品あり")
                      c_product += 1
              if(c_person>0):
                  count_person += 1
                  save_file_name = save_file_name + 'person_' + str(count_person).zfill(2) + ext              
              else:
                  count_product += 1
                  save_file_name = save_file_name + 'product_' + str(count_product).zfill(2) + ext
          else: #何も検知しなければ
              # print("画像に商品あり")
              count_product += 1
              save_file_name = save_file_name + 'product_' + str(count_product).zfill(2) + ext

          if True:
              # save_folder = os.path.join(
              #     vis_folder, time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
              # )
              # os.makedirs(save_folder, exist_ok=True)
              # save_file_name = os.path.join(save_folder, os.path.basename(image_name))
              # cv2.imwrite(save_file_name, result_image)
              cv2.imwrite(save_file_name, cv2.imread(image_name,-1))
          # ch = cv2.waitKey(0)
          # if ch == 27 or ch == ord("q") or ch == ord("Q"):
          #     print("image_demo break")
          #     break

#####修正点#####
def image_rename(exp, args, rename_image_path:str):
    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    file_name = os.path.join(exp.output_dir, args.experiment_name)
    os.makedirs(file_name, exist_ok=True)

    vis_folder = None
    if args.save_result:
        vis_folder = os.path.join(file_name, "vis_res")
        os.makedirs(vis_folder, exist_ok=True)

    if args.trt:
        args.device = "gpu"

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    model = exp.get_model()

    if args.device == "gpu":
        model.cuda()
        if args.fp16:
            model.half()  # to FP16
    model.eval()

    if not args.trt:
        if args.ckpt is None:
            ckpt_file = os.path.join(file_name, "best_ckpt.pth")
        else:
            ckpt_file = args.ckpt
        ckpt = torch.load(ckpt_file, map_location="cpu")
        # load the model state dict
        model.load_state_dict(ckpt["model"])

    if args.fuse:
        model = fuse_model(model)

    if args.trt:
        assert not args.fuse, "TensorRT model is not support model fusing!"
        trt_file = os.path.join(file_name, "model_trt.pth")
        assert os.path.exists(
            trt_file
        ), "TensorRT model is not found!\n Run python3 tools/trt.py first!"
        model.head.decode_in_inference = False
        decoder = model.head.decode_outputs
    else:
        trt_file = None
        decoder = None

    predictor = Predictor(
        model, exp, COCO_CLASSES, trt_file, decoder,
        args.device, args.fp16, args.legacy,
    )
    current_time = time.localtime()
    if args.demo == "image":
        image_demo(predictor, vis_folder, args.path, current_time, args.save_result, rename_image_path)

def get_tag_position(tag_path:str,tag_max_rate:float,tag_min_rate:float,tag_pos_list:list,bg_path:str):

  #タグの余白適正化
  tag_image = Image.open(tag_path)
  edges = find_edge(tag_image,"result")
  left,top,right,bottom = find_target(edges)
  tag_image = to_extrapolate(left,top,right,bottom,tag_image) #もし上下左右がくっついてなければ元画像をを拡張しておく

  tag_right, tag_bottom = tag_image.size
  tag_right = tag_right - 2 #なぜか2pixcelずれるため調整
  tag_bottom = tag_bottom - 2 #なぜか2pixcelずれるため調整
  tag_left = 0
  tag_top = 0

  edges = find_edge(tag_image,"result")
  left,top,right,bottom = find_target(edges)
  left,top,right,bottom,con_num = find_target_margin(tag_left,tag_top,tag_right,tag_bottom,left,top,right,bottom,0.03)
  tag_image = to_crop(tag_image,left, top, right, bottom)

  #背景のエッジ化
  bg_file = bg_path[bg_path.rfind('/')+1:]
  image = Image.open(bg_path)
  new_image = np.array(image, dtype=np.uint8)
  img = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
  blur = cv2.blur(img,(5,5))
  edges = cv2.Canny(blur,50,50)

  #背景の情報
  bg_type = int(bg_file[0:2])
  right = int(get_parameters(bg_file)[2])
  padding = int(right*0.02) #bg周りの余白
  h_ｔweak = int(right*0.02) #縦微調整分の余白
  w_ｔweak = int(right*0.02) #横微調整分の余白
  center = int(right - get_parameters(bg_file)[0])
  quarter = int(center + (right - center)/2)
  half = int(get_parameters(bg_file)[2]/2)
  if(bg_type >= 3): #正方形の組み合わせ用
    center = half

  #タグの情報
  tag_image_size = tag_image.copy()
  tag_image_size.thumbnail(size=(right, right))
  tag_original_width = tag_image_size.size[0]
  tag_original_height = tag_image_size.size[1]

  tag_rate = tag_max_rate #BGファイルに対するタグの面積の比率
  min_edges = 1000
  min_left = 0
  min_top = 0
  min_key = ''

  #タグの中に含まれるエッジの数が0になるまでtag_rateを下げる
  while min_edges > 0 and tag_rate >= tag_min_rate:
    #背景の面積からタグの縦横の比率を計算する
    tag_area = tag_original_height * tag_original_width
    bg_area = right * right
    tag_width = int(tag_original_width*math.sqrt(bg_area*tag_rate) / math.sqrt(tag_area))
    tag_height = int(tag_original_height*math.sqrt(bg_area*tag_rate) / math.sqrt(tag_area))
    tag_image_resize = tag_image.copy()
    if(tag_width>tag_height):
      tag_length = tag_width
    else:
      tag_length = tag_height
    tag_image_resize.thumbnail(size=(tag_length, tag_length))

    tag_height_padding = int(tag_image_resize.size[1]*0.1) #tag周りの余白
    tag_width_padding = int(tag_image_resize.size[0]*0.1) #tag周りの余白
    tag_height = tag_image_resize.size[1] + tag_height_padding
    tag_width = tag_image_resize.size[0] + tag_width_padding
    tag_height_half = int(tag_height/2)
    tag_width_half = int(tag_width/2)
    
    #タグを配置するポジション
    position = {
      7 : [quarter - tag_width_half , half - int(half*7/10) - tag_height_half], # 'quarter-quarter'

      13 : [center - tag_width_half , half - tag_height_half], # 'half-center'
      12 : [center - tag_width_half - w_ｔweak, half - tag_height_half], # 'half-center_left'
      14 : [center - tag_width_half + w_ｔweak, half - tag_height_half], # 'half-center_right'
      15 : [quarter - tag_width_half , half - tag_height_half], # 'half-quarter'

      9 : [center - tag_width_half , half - h_ｔweak - tag_height_half], # 'half_top-center'
      8 : [center - tag_width_half - w_ｔweak, half - h_ｔweak - tag_height_half], # 'half_top-center_left'
      10 : [center - tag_width_half + w_ｔweak, half - h_ｔweak - tag_height_half], # 'half_top-center_right'
      11 : [quarter - tag_width_half , half - h_ｔweak - tag_height_half], # 'half_top-quarter'

      17 : [center - tag_width_half , half + h_ｔweak - tag_height_half], # 'half_bottom-center'
      16 : [center - tag_width_half - w_ｔweak, half + h_ｔweak - tag_height_half], # 'half_bottom-center_left'
      18 : [center - tag_width_half + w_ｔweak, half + h_ｔweak - tag_height_half], # 'half_bottom-center_right'
      19 : [quarter - tag_width_half , half + h_ｔweak - tag_height_half], # 'half_bottom-quarter'

      1 : [padding , padding], # 'top-left'
      5 : [quarter - tag_width_half , padding], # 'top-quarter' 
      3 : [center - tag_width_half , padding], # 'top-center'
      2 : [center - tag_width_half - w_ｔweak, padding], # 'top-center_left'
      4 : [center - tag_width_half + w_ｔweak, padding], # 'top-center_right'
      6 : [right - tag_width - padding , padding], # 'top-right'

      20 : [padding , right - tag_height - padding], # 'bottom-left'
      24 : [quarter - tag_width_half , right - tag_height - padding], # 'bottom-quarter'
      22 : [center - tag_width_half , right - tag_height - padding], # 'bottom-center'
      21 : [center - tag_width_half - w_ｔweak, right - tag_height - padding], # 'bottom-center_left'
      23 : [center - tag_width_half + w_ｔweak, right - tag_height - padding], # 'bottom-center_right'
      25 : [right - tag_width - padding , right - tag_height - padding], # 'bottom-right'

    }

    # ポジションの指定がある場合は指定
    if(len(tag_pos_list)>0):
      tmp_position = {}
      for tag_pos in tag_pos_list:
        tmp_position[tag_pos] = position[tag_pos]
      position = tmp_position

    # 画面からタグがはみ出るポジションは削除
    for key in position:
      if(position[key][0] + tag_width + padding > right):
        position[key][0] = right - tag_width - padding
      elif(position[key][0] - padding < 0):
        position[key][0] = 0 +  padding
      elif(position[key][1] + tag_height + padding > right):
        position[key][0] = right - tag_height - padding
      elif(position[key][0] - padding < 0):
        position[key][0] = 0 + padding

    #全てのポジションにタグを置いた場合、エッジ数が最小になるタグ情報
    for key in position:
      # print("min_edges:" + str(min_edges)+ " " +str(edges[position[key][1]:position[key][1]+tag_height,position[key][0]:position[key][0]+tag_width].nonzero()[0].size) + " position[key][0]:" + str(position[key][0])+ " position[key][1]:" + str(position[key][1]))
      if(edges[position[key][1]:position[key][1]+tag_height,position[key][0]:position[key][0]+tag_width].nonzero()[0].size < min_edges):
        min_edges = edges[position[key][1]:position[key][1]+tag_height,position[key][0]:position[key][0]+tag_width].nonzero()[0].size
        min_left = position[key][0]
        min_top = position[key][1]
        min_key = key
        #TODO テスト表示用
        # rec = cv2.rectangle(np.ascontiguousarray(image), (position[key][0], position[key][1]), (position[key][0] + int(tag_width), position[key][1] + int(tag_height)), (0, 255, 0), thickness=-1)

    tag_rate -= 0.001

  #TODO テスト表示用
  # out_dir = "./image/result_image/"
  # path_pos = out_dir + bg_file[:bg_file.rfind('.')] + "_mask" + '.png'
  # cv2.imwrite(path_pos, rec)
  # path_edges = out_dir + bg_file[:bg_file.rfind('.')] + "_edges" + '.png'
  # cv2.imwrite(path_edges, edges)  

  #タグのカラー変更処理
  bg_color = [image.getpixel((min_left,min_top)),image.getpixel((min_left+tag_width,min_top)),image.getpixel((min_left,min_top+tag_height)),image.getpixel((min_left+tag_width,min_top+tag_height))]
  color_count = 0
  for color in bg_color:
    if(color[0]<50 and color[2]<50 and color[2]<50):
      color_count += 1
  tag_image_resize_colored = np.array(tag_image, dtype=np.uint8)
  tag_image_resize_colored = cv2.cvtColor(tag_image_resize_colored, cv2.COLOR_RGBA2BGRA)
  if(color_count > 0): #タグの四隅のどこかが黒っぽかったら
    tag_image_resize_colored[:, :, 3] = np.where(np.all(tag_image_resize_colored >= 240, axis=-1), 50, 255)  # 半透明な背景にする
  else:
    tag_image_resize_colored[:, :, 3] = np.where(np.all(tag_image_resize_colored >= 240, axis=-1), 0, 255)  # 背景を透過させる
  tag_image_resize_colored = cv2pil(tag_image_resize_colored)
  tag_image_resize_colored.thumbnail(size=(tag_length, tag_length))

  # print("tag resize " + str(tag_original_width) + ":" + str(tag_original_height) + " => " + str(tag_width) + ":" + str(tag_height))
  # print("tag rate " + str(tag_rate))
  # print("最小エッジ数は " + str(min_edges))
  # print("タグのleftは " + str(min_left))
  # print("タグのtopは " + str(min_top))
  # print("タグのポジションは " + str(min_key))
  # print("タグの面積は " + str(tag_height*tag_width))
  
  return tag_image_resize_colored,int(min_left+tag_width_padding/2),int(min_top+tag_height_padding/2)

def image_resize(person_frame_percent:float,product_frame_percent:float,pixcel:int,rename_image_path:str,resize_image_path:str):
  ##personファイルのリサイズ処理
  person_x_list = [int(pixcel*2/5),int(pixcel/2),int(pixcel*3/5),int(pixcel*2/3)]
  for filename in match_file(rename_image_path,'person'):
    for x in person_x_list:
      y = pixcel
      image_type = filename.split('_')[0]
      input = rename_image_path + '/' + filename
      output = resize_image_path + '/' + 'person-' + filename[:filename.rfind('.')][-2:] + '_' + 'w' + str(x) + 'h' + str(y) + 'px' + str(pixcel) + '.png'
      aspect_margin = get_resize_image(input,x,y,person_frame_percent,image_type,output)
      if(aspect_margin >= 0):
        break

  ##productファイルのリサイズ処理
  product_x_list = []
  product_y_list = []
  if(len(match_file(resize_image_path,'person'))>0):
    product_y_list = [int(pixcel),int(pixcel/2)]
    product_x_list = set([ pixcel - get_parameters(person)[0] for person in match_file(resize_image_path,'person') ])
  else:
    product_x_list = [int(pixcel),int(pixcel/2)]
    product_y_list = [int(pixcel),int(pixcel/2)]

  for filename in match_file(rename_image_path,'product'):
    for x in product_x_list:
      for y in product_y_list:
        if(len(match_file(resize_image_path,'person')) > 0 or (len(match_file(resize_image_path,'person')) == 0 and x==y)): #psersonなし
          image_type = filename.split('_')[0]
          input = rename_image_path + '/' + filename
          output = resize_image_path + '/' + 'product-' + filename[:filename.rfind('.')][-2:] + '_' + 'w' + str(x) + 'h' + str(y) + 'px' + str(pixcel) + '.png'
          aspect_margin = get_resize_image(input,x,y,product_frame_percent,image_type,output)

def image_composition(output_max_num:int,resize_image_path:str,result_image_path:str):
  max_height = get_parameters(match_file(resize_image_path,'product')[0])[2] #pxを最大の高さとする
  min_height = str(int(int(max_height)/2))
  min_width = str(int(int(max_height)/2))
  im_ext = ".png"

  ##人ありのパターン（Type01、Type02）
  if(len(match_file(resize_image_path,'person')) > 0):
    for product_x in set([ get_parameters(product)[0] for product in match_file(resize_image_path,'product') ]): # productのユニークなxだけ抽出　 降順でsortすることで一番幅の狭い人の画像から使う
      min_width = max_height - product_x #personのwidth

      ###縦の合成処理
      vconcat_list = sorted(match_file(resize_image_path,str("w" + str(product_x)) + "h" + str(min_height))) # sortすることで最初の画像を優先して先頭へ使う
      for pair in list(itertools.combinations(vconcat_list, 2)):
        path_im_1 = resize_image_path + "/" + pair[0]
        id_im_1 = get_parameters(pair[0])[3]
        path_im_2 = resize_image_path + "/" + pair[1]
        id_im_2 = get_parameters(pair[1])[3]
        path_im_v = resize_image_path + "/" + 'cproduct-' + id_im_1 + id_im_2 + '_' + 'w' + str(product_x) + 'h' + str(max_height) + 'px' + str(max_height) + im_ext
        im_v = cv2.vconcat([cv2.imread(path_im_1), cv2.imread(path_im_2)])
        cv2.imwrite(path_im_v, im_v)

    ###横の合成処理
    hconcat_list = match_file(resize_image_path,"product") #対象のwidthで高さがmaxのリスト
    hconcat_list = [p for p in hconcat_list if p[:p.find("-")] == "product" or p[:p.find("-")] == "cproduct"] #そこから更にproductのみを抽出
    for im, im_person in sorted(itertools.product(hconcat_list, sorted(match_file(resize_image_path,'person')))):
        if(get_parameters(im)[1] == get_parameters(im)[2] and get_parameters(im)[0] == (get_parameters(im_person)[2] - get_parameters(im_person)[0])):
          path_im = resize_image_path + "/" + im
          id_im = get_parameters(im)[3]
          path_im_person = resize_image_path + "/" + im_person
          id_im_person = get_parameters(im_person)[3]
          if len(id_im) == 2:
            path_im_h = result_image_path + '/' + '02result-' + id_im + id_im_person + '_' + 'w' + str(get_parameters(im)[0]) + 'h' + str(max_height) + 'px' + str(max_height) + im_ext
          else:
            path_im_h = result_image_path + '/' + '01result-' + id_im + id_im_person + '_' + 'w' + str(get_parameters(im)[0]) + 'h' + str(max_height) + 'px' + str(max_height) + im_ext
          im_h = cv2.hconcat([cv2.imread(path_im_person), cv2.imread(path_im)])
          if output_max_num > 0 :
            cv2.imwrite(path_im_h, im_h)
            output_max_num -= 1
          else:
            break

  ##人なしのパターン（Type03、Type04）
  else:
    ###Type03
    min_width = str(int(int(max_height)/2))
    vconcat_list = sorted(match_file(resize_image_path,str("w" + str(min_width)) + "h" + str(min_height))) #1/2 * 1/2のリスト sortすることで最初の画像を優先して先頭へ使う
    for comb in list(itertools.combinations(vconcat_list, 4)):
      path_im_1 = resize_image_path + "/" + comb[0]
      id_im_1 = get_parameters(comb[0])[3]
      path_im_2 = resize_image_path + "/" + comb[1]
      id_im_2 = get_parameters(comb[1])[3]
      path_im_3 = resize_image_path + "/" + comb[2]
      id_im_3 = get_parameters(comb[2])[3]
      path_im_4 = resize_image_path + "/" + comb[3]
      id_im_4 = get_parameters(comb[3])[3]
      path_im_h = result_image_path + '/' + '03result-' + id_im_1 + id_im_2 + id_im_3 + id_im_4 + '_' + 'w' + str(min_width) + 'h' + str(max_height) + 'px' + str(max_height) + im_ext
      im_v_01 = cv2.vconcat([cv2.imread(path_im_1), cv2.imread(path_im_2)])
      im_v_02 = cv2.vconcat([cv2.imread(path_im_3), cv2.imread(path_im_4)])
      im_h = cv2.hconcat([im_v_01,im_v_02])
      if output_max_num > 0 :
        cv2.imwrite(path_im_h, im_h)
        output_max_num -= 1
      else:
        break

    ###Type04
    vconcat_list = match_file(resize_image_path,str("w" + str(max_height)) + "h" + str(max_height)) #1/1 * 1/1のリスト
    for im in vconcat_list:
      path_im = resize_image_path + "/" + im
      id_im = get_parameters(im)[3]
      path_im_h = result_image_path + '/' + '04result-' + id_im + '_' + 'w' + str(min_width) + 'h' + str(max_height) + 'px' + str(max_height) + im_ext
      if output_max_num > 0 :
        cv2.imwrite(path_im_h, cv2.imread(path_im))
        output_max_num -= 1
      else:
        break

def image_add_tag(tag_list:list,result_image_path):
  for tag_path in tag_list:
    for bg_file in match_file(result_image_path,'t'):
      bg_path = result_image_path + "/" + bg_file
      tag_image_resize , tag_position_left , tag_position_top = get_tag_position(tag_path[0],tag_path[1],tag_path[2],tag_path[3],bg_path)
      bg1 = Image.open(bg_path).convert('RGBA')
      img_clear = Image.new("RGBA", bg1.size, (255, 255, 255, 0))
      img_clear.paste(tag_image_resize, (tag_position_left, tag_position_top))
      bg1 = Image.alpha_composite(bg1, img_clear)
      bg1_out_path = result_image_path + "/" +  bg_file
      bg1.save(bg1_out_path,'png', quality=100)

def image_add_frame(frame_aseet_path:str,result_image_path:str):
  max_height = get_parameters(match_file(result_image_path,'result')[0])[2] #pxを最大の高さとする
  frame_image = cv2.imread(frame_aseet_path, -1)
  frame_image[:, :, 3] = np.where(np.all(frame_image == 255, axis=-1), 0, 255)  # 白色のみTrueを返し、Alphaを0にする
  frame_image = cv2pil(frame_image)
  frame_image.thumbnail(size=(int(max_height), int(max_height)))
  for bg_file in match_file(result_image_path,'t'):
    bg_path = result_image_path + "/" + bg_file
  
    bg = Image.open(bg_path).convert('RGBA')
    img_clear = Image.new("RGBA", (int(max_height), int(max_height)), (255, 255, 255, 0))
    img_clear.paste(frame_image, (0, 0))
    bg = Image.alpha_composite(bg, img_clear)
    bg_out_path = result_image_path + "/" +  bg_file
    bg.save(bg_out_path,'png', quality=100)

def the_thread(window,sorce_image_path,result_image_path,pixcel,output_max_num,person_frame_percent,product_frame_percent,frame_file_name,tag1_file_name,tag1_percent_min,tag1_percent_max,tag1_position,tag2_file_name,tag2_percent_min,tag2_percent_max,tag2_position,tag3_file_name,tag3_percent_min,tag3_percent_max,tag3_position):
  try:
    #作業フォルダ一覧作成
    files = os.listdir(sorce_image_path)
    dir_list = [f for f in files if os.path.isdir(os.path.join(sorce_image_path, f))]
    print("指定されたフォルダの中に画像合成用のフォルダがなかったため作業を終了します") if len(dir_list) == 0 else print('作業開始')
    complete_time = 0
    counter = 1
    for dir in dir_list:
      if stop_flg : ##停止フラグがONであればループから抜ける
        break
      start = time.time() 
      sorce_image_dir = sorce_image_path + "/" + dir
      result_image_dir = result_image_path + "/" + dir
      print("### " + sorce_image_dir + " フォルダを作業開始 ###")

      #作業画像の削除
      delete_files([rename_image_path,resize_image_path])

      #合成前のGUI表示用画像作成および通知
      img = Image.new("RGB", (pixcel, pixcel*20), (227, 227, 227))
      img.save(display_image_path + '/' + 'display_before_image.png')

      img_files = get_image_list(sorce_image_dir) #見やすくするようにタグファイルの順番入れ替え
      if(tag1_file_name != "" and os.path.exists(str(sorce_image_dir + "/" + tag1_file_name))):
        img_files = [filename for filename in img_files if not tag1_file_name in filename]
        img_files.append(str(sorce_image_dir + "/" + tag1_file_name))
      if(tag2_file_name != "" and os.path.exists(str(sorce_image_dir + "/" + tag2_file_name))):
        img_files = [filename for filename in img_files if not tag2_file_name in filename]
        img_files.append(str(sorce_image_dir + "/" + tag2_file_name))
      if(tag3_file_name != "" and os.path.exists(str(sorce_image_dir + "/" + tag3_file_name))):
        img_files = [filename for filename in img_files if not tag3_file_name in filename]
        img_files.append(str(sorce_image_dir + "/" + tag3_file_name))
      if(frame_file_name != "" and os.path.exists(str(sorce_image_dir + "/" + frame_file_name))):
        img_files = [filename for filename in img_files if not frame_file_name in filename]
        img_files.append(str(sorce_image_dir + "/" + frame_file_name))
      img_files.reverse()
      for filename in img_files:
        path_im_1 = display_image_path + '/' + 'display_before_image.png'
        path_im_2 = filename
        im_1_height, im_1_width = cv2.imread(path_im_1).shape[:2]
        h, w = cv2.imread(path_im_2).shape[:2]
        #ここで素材画像（path_im_2）のリサイズ処理が必要
        height = int(h * (im_1_width / w))
        resize_img = cv2.resize(cv2.imread(path_im_2), dsize=(im_1_width, height))
        im_v = cv2.vconcat([resize_img, cv2.imread(path_im_1)])
        im_v = im_v[0:im_1_width*20,0:im_1_width]
        cv2.imwrite(path_im_1, im_v)
        
      window.write_event_value('-THREAD-', 'before')
      window.write_event_value('-THREAD-', 'prog00')
      #画像のリネーム処理
      print("#画像のリネーム処理")
      os.chdir(yolox_path)
      args = make_parser(sorce_image_dir).parse_args()
      exp = get_exp(None, args.name)
      image_rename(exp, args, rename_image_path)
      window.write_event_value('-THREAD-', 'prog01')

      #画像のリサイズ処理
      print("#画像のリサイズ処理")
      os.chdir(home_path)
      image_resize(person_frame_percent,product_frame_percent,pixcel,rename_image_path,resize_image_path)
      window.write_event_value('-THREAD-', 'prog02')

      #画像合成処理
      print("#画像合成処理")
      os.chdir(home_path)
      os.makedirs(result_image_dir, exist_ok=True)
      delete_files([result_image_dir]) #何回か試すとゴミが溜まってしまうため、一度作業前に削除
      try:
        image_composition(output_max_num,resize_image_path,result_image_dir)
      except IndexError:
        print('エラー：画像合成を満たす基準の画像がなかったため合成できませんでした。')
        delete_files([rename_image_path,resize_image_path])
        window.write_event_value('-THREAD-', int((counter / len(dir_list))*100))
        counter += 1
        continue
      window.write_event_value('-THREAD-', 'prog03')

      #タグ合成処理
      print("#タグ合成処理")
      os.chdir(home_path)
      tag_list = []
      if(os.path.isfile(str(sorce_image_dir + "/" + tag1_file_name))):
        tag_list.append([str(sorce_image_dir + "/" + tag1_file_name),tag1_percent_max,tag1_percent_min,tag1_position])
      if(os.path.isfile(str(sorce_image_dir + "/" + tag2_file_name))):
        tag_list.append([str(sorce_image_dir + "/" + tag2_file_name),tag2_percent_max,tag2_percent_min,tag2_position])
      if(os.path.isfile(str(sorce_image_dir + "/" + tag3_file_name))):
        tag_list.append([str(sorce_image_dir + "/" + tag3_file_name),tag3_percent_max,tag3_percent_min,tag3_position])
      image_add_tag(tag_list,result_image_dir)
      window.write_event_value('-THREAD-', 'prog04')

      #フレーム合成処理
      print("#フレーム合成処理")
      os.chdir(home_path)
      frame_aseet_path = sorce_image_dir + "/" + frame_file_name
      if(os.path.isfile(frame_aseet_path)):
        image_add_frame(frame_aseet_path,result_image_dir)

      window.write_event_value('-THREAD-', 'prog05')

      end = time.time()
      elapsed_time = end - start
      complete_time += elapsed_time
      print("### 作成時間 " + str(elapsed_time) + " 秒 ###")

      #合成後GUI表示用画像作成および通知
      img = Image.new("RGB", (pixcel, pixcel*20), (227, 227, 227))
      img.save(display_image_path + '/' + 'display_after_image.png')
      for filename in get_image_list(result_image_dir):
        path_im_1 = display_image_path + '/' + 'display_after_image.png'
        path_im_2 = filename
        h, w = cv2.imread(path_im_1).shape[:2]
        im_v = cv2.vconcat([cv2.imread(path_im_2), cv2.imread(path_im_1)])
        im_v = im_v[0:w*20,0:w]
        cv2.imwrite(path_im_1, im_v)
        
      window.write_event_value('-THREAD-', 'after')
      window.write_event_value('-THREAD-', int((counter / len(dir_list))*100))
      counter += 1
      
    print("### 合計時間 " + str(complete_time) + " 秒 ###")
    window.write_event_value('-THREAD-', 'end')
    return "OK"
  except FileNotFoundError:
    print(sorce_image_path)
    print('エラー：指定されたフォルダが見つかりませんでした')
    print('再度フォルダの指定を確認するか、新たにフォルダを作成して下さい')
    delete_files([rename_image_path,resize_image_path])
    window.write_event_value('-THREAD-', 'end')
    return "NG"

pip.main(['install','PySimpleGUI'])
pip.main(['install','threading'])
importlib.reload(site)

import PySimpleGUI as sg
import threading
import re

global stop_flg
stop_flg = False

def get_display_data(f, width=261):
  img = cv2.imread(f)
  h, w = img.shape[:2]
  height = int(h * (width / w))
  resize_img = cv2.resize(img, dsize=(width, height))
  return cv2.imencode('.png', resize_img)[1].tobytes()

#GUI用変数定義
THREAD_EVENT = '-THREAD-'
img = Image.new("RGB", (pixcel, pixcel*20), (227, 227, 227))
img.save(display_image_path + '/' + 'display_after_image.png')
display_after_image = get_display_data(display_image_path + '/' + 'display_after_image.png')
img.save(display_image_path + '/' + 'display_before_image.png')
display_before_image = get_display_data(display_image_path + '/' + 'display_before_image.png')

#GUI設定
sg.theme('LightGray6')
frame1_col11 = [
    [
      sg.Text("　合成素材フォルダ", font=('メイリオ',12)),
    ],
]
frame1_col12 = [
    [
      sg.FolderBrowse(button_text='フォルダ選択', initial_folder=home_path, font=('メイリオ',10), size=(8,1), key="-SOURCEDIR-",target='-SOURCEDIRPATH-'),
    ],
]
frame1_col13 = [
    [
      sg.InputText(default_text = sorce_image_path, key='-SOURCEDIRPATH-', size=(60,1), enable_events=True), 
    ],
]

frame1_col21 = [
    [
      sg.Text("　合成結果フォルダ", font=('メイリオ',12)),
    ],
]
frame1_col22 = [
    [
      sg.FolderBrowse(button_text='フォルダ選択', initial_folder=home_path, font=('メイリオ',10), size=(8,1), key="-RESULTDIR-",target='-RESULTDIRPATH-'),
    ],
]
frame1_col23 = [
    [
      sg.InputText(default_text = result_image_path, key='-RESULTDIRPATH-', size=(60,1), enable_events=True), 
    ],
]
frame1_col31 = [
    [
      sg.Text("　合成画像サイズ（正方形）", font=('メイリオ',12)),
    ],
    [
      sg.Text("　人画像の周りの余白", font=('メイリオ',12)),
    ]
]
frame1_col32 = [
    [
      sg.InputText(default_text = str(pixcel), size=(4,1), key='-PIXCEL-', enable_events=True), 
      sg.Text("px (200-1000)", font=('メイリオ',12)),
    ],
    [
      sg.InputText(default_text = str(person_frame_percent*100), size=(4,1), key='-PERSON_FRAME_PERCENT-', enable_events=True), 
      sg.Text("% (0.0-10.0)", font=('メイリオ',12)),
    ],
]
frame1_col33 = [
    [
      sg.Text("　合成画像の最大組み合わせ数", font=('メイリオ',12)),
    ],
    [
      sg.Text("　商品画像の周りの余白", font=('メイリオ',12)),
    ],
]
frame1_col34 = [
    [
      sg.InputText(default_text = str(output_max_num), size=(4,1), key='-OUTPUT_MAX_NUM-', enable_events=True), 
      sg.Text(" (1-10)", font=('メイリオ',12)),
    ],
    [
      sg.InputText(default_text = str(product_frame_percent*100), size=(4,1), key='-PRODUCT_FRAME_PERCENT-', enable_events=True), 
      sg.Text("% (0.0-10.0)", font=('メイリオ',12)),
    ],
]
frame1_col41 = [
    [
      sg.Text("　装飾するフレームファイル名", font=('メイリオ',12)),
    ],
]
frame1_col42 = [
    [
      sg.InputText(default_text = 'frame.png', size=(10,1), key='-FRAME_FILE_NAME-', enable_events=True), 
    ],
]

frame1_col511 = [
    [
      sg.Text("　挿入するファイル名１", font=('メイリオ',12)),
    ],
    [
      sg.Text("　挿入する位置の候補", font=('メイリオ',12)),
    ],
]
frame1_col512 = [
    [
      sg.InputText(default_text = 'logo.png', size=(10,1), key='-TAG1_FILE_NAME-', enable_events=True), 
      sg.Text("　合成画像に対する割合", font=('メイリオ',12)),
      sg.InputText(default_text = str(tag1_percent_min*100), size=(4,1), key='-TAG1_PERENT_MIN-', enable_events=True), 
      sg.Text("%", font=('メイリオ',12)),
      sg.Text("〜", font=('メイリオ',12)),
      sg.InputText(default_text = str(tag1_percent_max*100), size=(4,1), key='-TAG1_PERENT_MAX-', enable_events=True), 
      sg.Text("% (1.0-5.0)", font=('メイリオ',12)),
    ],
    [
      sg.InputText(default_text = '7,13,12,14,15,9,8,10,11,17,16,18,19,1,5,3,2,4,6,20,24,22,21,23,25', size=(40,1), key='-TAG1_POSITION-', enable_events=True), 
      sg.Text("優先する箇所からカンマ(,)区切りで指定", font=('メイリオ',10)),
    ],
]
frame1_col521 = [
    [
      sg.Text("　挿入するファイル名２", font=('メイリオ',12)),
    ],
    [
      sg.Text("　挿入する位置の候補", font=('メイリオ',12)),
    ],
]
frame1_col522 = [
    [
      sg.InputText(default_text = 'shop.png', size=(10,1), key='-TAG2_FILE_NAME-', enable_events=True), 
      sg.Text("　合成画像に対する割合", font=('メイリオ',12)),
      sg.InputText(default_text = str(tag2_percent_min*100), size=(4,1), key='-TAG2_PERENT_MIN-', enable_events=True), 
      sg.Text("%", font=('メイリオ',12)),
      sg.Text("〜", font=('メイリオ',12)),
      sg.InputText(default_text = str(tag2_percent_max*100), size=(4,1), key='-TAG2_PERENT_MAX-', enable_events=True), 
      sg.Text("% (1.0-5.0)", font=('メイリオ',12)),
    ],
    [
      sg.InputText(default_text = '1,6,20,25', size=(40,1), key='-TAG2_POSITION-', enable_events=True), 
      sg.Text("優先する箇所からカンマ(,)区切りで指定", font=('メイリオ',10)),
    ],
]
frame1_col531 = [
    [
      sg.Text("　挿入するファイル名３", font=('メイリオ',12)),
    ],
    [
      sg.Text("　挿入する位置の候補", font=('メイリオ',12)),
    ],
]
frame1_col532 = [
    [
      sg.InputText(size=(10,1), key='-TAG3_FILE_NAME-', enable_events=True), 
      sg.Text("　合成画像に対する割合", font=('メイリオ',12)),
      sg.InputText(size=(4,1), key='-TAG3_PERENT_MIN-', enable_events=True), 
      sg.Text("%", font=('メイリオ',12)),
      sg.Text("〜", font=('メイリオ',12)),
      sg.InputText(size=(4,1), key='-TAG3_PERENT_MAX-', enable_events=True), 
      sg.Text("% (1.0-5.0)", font=('メイリオ',12)),
    ],
    [
      sg.InputText(size=(40,1), key='-TAG3_POSITION-', enable_events=True), 
      sg.Text("優先する箇所からカンマ(,)区切りで指定", font=('メイリオ',10)),
    ],
]
frame1_col541 = [
    [
      sg.Text("　挿入位置　　　　　　", font=('メイリオ',12)),
    ],
]
frame1_col542 = [
    [
      sg.Image(source='./asset/gui/setumei.png'),
    ],
]
frame1 = sg.Frame('',
    [
      [
          sg.Text("1.合成する画像が保存されているフォルダを指定して下さい。", font=('メイリオ',12), pad=((5,5),(5,10))),
      ],
      [   
          sg.Column(frame1_col11), sg.Column(frame1_col12), sg.Column(frame1_col13),
      ],
      [
          sg.Text("2.合成した画像を保存するフォルダを指定して下さい。", font=('メイリオ',12), pad=(5,10)),
      ],
      [   
          sg.Column(frame1_col21), sg.Column(frame1_col22), sg.Column(frame1_col23),
      ],
      [
          sg.Text("3.各種設定値を入力して下さい", font=('メイリオ',12), pad=(5,10)),
      ],
      [   
          sg.Column(frame1_col31), sg.Column(frame1_col32), sg.Column(frame1_col33), sg.Column(frame1_col34),
      ],
      [
          sg.Text("4.合成画像を装飾するフレーム画像", font=('メイリオ',12), pad=(5,10)),
      ],
      [   
          sg.Column(frame1_col41), sg.Column(frame1_col42)
      ],
      [
          sg.Text("5.合成画像に挿入するロゴやショップ名などの画像（最大３つまで）", font=('メイリオ',12), pad=(5,10)),
          sg.Text("　※上位に指定したファイルから優先的に挿入されます", font=('メイリオ',10), pad=(5,10)),
      ],
      [   
          sg.Column(frame1_col511), sg.Column(frame1_col512)
      ],
      [   
          sg.Column(frame1_col521), sg.Column(frame1_col522)
      ],
      [   
          sg.Column(frame1_col531), sg.Column(frame1_col532)
      ],
      [   
          sg.Column(frame1_col541), sg.Column(frame1_col542),
      ],
    ], size=(650, 800),border_width=0
)

frame2_col11 = [
      [sg.Submit(button_text='実行する', font=('メイリオ',20),size=(15,1),button_color=('white', 'green'),key='-START_BUTTON-'),]
]
frame2_col12 = [
      [sg.Submit(button_text='停止する', font=('メイリオ',20),size=(15,1),button_color=('white', 'red'),key='-END_BUTTON-',disabled=True),]
]
frame2 = sg.Frame('',
    [      
      [   
          sg.Column(frame2_col11, justification="c"),
          sg.Column(frame2_col12, justification="c")
      ],
    ], size=(600, 45),border_width=0
)
frame3_col11 = [
    [
      sg.Image(data=display_before_image,key='-DISPLAY_BEFORE_IMAGE-')
    ],
]
frame3 = sg.Frame('合成前',
    [      
      [   
          sg.Column(frame3_col11, justification="c", scrollable=True, vertical_scroll_only=True, size=(300, 800))
      ],
    ], size=(300, 600) #幅,高さ
)
frame4_col11 = [
    [
      sg.Image(data=display_after_image,key='-DISPLAY_AFTER_IMAGE-')
    ],
]
frame4 = sg.Frame('合成後',
    [      
      [   
          sg.Column(frame4_col11, justification="c", scrollable=True, vertical_scroll_only=True, size=(300, 800))
      ],
    ], size=(300, 600) #幅,高さ
)

frame5_col11 = [
      [sg.ProgressBar(100, orientation='h', size=(65,15), key='-PROG_UNIT-', bar_color=('#39B60A','#fff'), pad=(10,1))]
]

frame5_col21 = [
      [sg.ProgressBar(100, orientation='h', size=(65,15), key='-PROG_GROUP-', bar_color=('#39B60A','#fff'), pad=(10,0))]
]

frame5 = sg.Frame('',
    [      
      [   
          sg.Column(frame5_col11, justification="c", size=(600, 20))
      ],
      [   
          sg.Column(frame5_col21, justification="c", size=(600, 20))
      ],
    ], size=(600, 50),border_width=0
)

frame_in_column2 = sg.Column([
    [frame3, frame4],
])
frame_in_column = sg.Column([[frame2],
                             [frame_in_column2],
                             [frame5]
                             ])

t1 = sg.Tab('実行画面' ,[[frame_in_column]], key='-TAB1-')
t2 = sg.Tab('設定画面' ,[[frame1]], key='-TAB2-')
layout = [[sg.TabGroup ([[t1 ,t2]], key='-TABGROUP-')]
]
window = sg.Window('画像クリエイター', layout, resizable=True)

#GUI表示実行部分
while True:
    # ウィンドウ表示
    event, values = window.read()

    #クローズボタンの処理
    if event is None:
      print('exit')
      break

    #入力制限処理
    if event == '-SOURCEDIRPATH-':
      window['-SOURCEDIRPATH-'].update(background_color='#4A5C78')
    if event == '-RESULTDIRPATH-':
      window['-RESULTDIRPATH-'].update(background_color='#4A5C78')
    if event == '-PIXCEL-' and values['-PIXCEL-'] and values['-PIXCEL-'][-1] not in ('0123456789'):
      window['-PIXCEL-'].update(values['-PIXCEL-'][:-1])      
    if event == '-PIXCEL-' and values['-PIXCEL-'] and values['-PIXCEL-'][-1] in ('0123456789'):
      window['-PIXCEL-'].update(background_color='#4A5C78')
    if event == '-OUTPUT_MAX_NUM-' and values['-OUTPUT_MAX_NUM-'] and values['-OUTPUT_MAX_NUM-'][-1] not in ('0123456789'):
      window['-OUTPUT_MAX_NUM-'].update(values['-OUTPUT_MAX_NUM-'][:-1])      
    if event == '-OUTPUT_MAX_NUM-' and values['-OUTPUT_MAX_NUM-'] and values['-OUTPUT_MAX_NUM-'][-1] in ('0123456789'):
      window['-OUTPUT_MAX_NUM-'].update(background_color='#4A5C78')
    if event == '-PERSON_FRAME_PERCENT-' and values['-PERSON_FRAME_PERCENT-'] and values['-PERSON_FRAME_PERCENT-'][-1] not in ('0123456789.'):
      window['-PERSON_FRAME_PERCENT-'].update(values['-PERSON_FRAME_PERCENT-'][:-1])      
    if event == '-PERSON_FRAME_PERCENT-' and values['-PERSON_FRAME_PERCENT-'] and values['-PERSON_FRAME_PERCENT-'][-1] in ('0123456789.'):
      window['-PERSON_FRAME_PERCENT-'].update(background_color='#4A5C78')
    if event == '-PRODUCT_FRAME_PERCENT-' and values['-PRODUCT_FRAME_PERCENT-'] and values['-PRODUCT_FRAME_PERCENT-'][-1] not in ('0123456789.'):
      window['-PRODUCT_FRAME_PERCENT-'].update(values['-PRODUCT_FRAME_PERCENT-'][:-1])      
    if event == '-PRODUCT_FRAME_PERCENT-' and values['-PRODUCT_FRAME_PERCENT-'] and values['-PRODUCT_FRAME_PERCENT-'][-1] in ('0123456789.'):
      window['-PRODUCT_FRAME_PERCENT-'].update(background_color='#4A5C78')
    if event == '-FRAME_FILE_NAME-' and values['-FRAME_FILE_NAME-'] and not re.search(r'[0-9a-zA-Z\.]', values['-FRAME_FILE_NAME-'][-1]):
        window['-FRAME_FILE_NAME-'].update(values['-FRAME_FILE_NAME-'][:-1])      
    if event == '-FRAME_FILE_NAME-' and values['-FRAME_FILE_NAME-'] and re.search(r'[0-9a-zA-Z\.]', values['-FRAME_FILE_NAME-'][-1]):
      window['-FRAME_FILE_NAME-'].update(background_color='#4A5C78')
    if event == '-TAG1_FILE_NAME-' and values['-TAG1_FILE_NAME-'] and not re.search(r'[0-9a-zA-Z\.]', values['-FRAME_FILE_NAME-'][-1]):
        window['-TAG1_FILE_NAME-'].update(values['-TAG1_FILE_NAME-'][:-1])
    if event == '-TAG1_FILE_NAME-' and values['-TAG1_FILE_NAME-'] and re.search(r'[0-9a-zA-Z\.]', values['-FRAME_FILE_NAME-'][-1]):
      window['-TAG1_FILE_NAME-'].update(background_color='#4A5C78')
    if event == '-TAG1_PERENT_MIN-' and values['-TAG1_PERENT_MIN-'] and values['-TAG1_PERENT_MIN-'][-1] not in ('0123456789.'):
      window['-TAG1_PERENT_MIN-'].update(values['-TAG1_PERENT_MIN-'][:-1])      
    if event == '-TAG1_PERENT_MIN-' and values['-TAG1_PERENT_MIN-'] and values['-TAG1_PERENT_MIN-'][-1] in ('0123456789.'):
      window['-TAG1_PERENT_MIN-'].update(background_color='#4A5C78')
    if event == '-TAG1_PERENT_MAX-' and values['-TAG1_PERENT_MAX-'] and values['-TAG1_PERENT_MAX-'][-1] not in ('0123456789.'):
      window['-TAG1_PERENT_MAX-'].update(values['-TAG1_PERENT_MAX-'][:-1])      
    if event == '-TAG1_PERENT_MAX-' and values['-TAG1_PERENT_MAX-'] and values['-TAG1_PERENT_MAX-'][-1] in ('0123456789.'):
      window['-TAG1_PERENT_MAX-'].update(background_color='#4A5C78')
    if event == '-TAG1_POSITION-' and values['-TAG1_POSITION-'] and values['-TAG1_POSITION-'][-1] not in ('0123456789,'):
      window['-TAG1_POSITION-'].update(values['-TAG1_POSITION-'][:-1])      
    if event == '-TAG1_POSITION-' and values['-TAG1_POSITION-'] and values['-TAG1_POSITION-'][-1] in ('0123456789,'):
      window['-TAG1_POSITION-'].update(background_color='#4A5C78')
    if event == '-TAG2_FILE_NAME-' and values['-TAG2_FILE_NAME-'] and not re.search(r'[0-9a-zA-Z\.]', values['-FRAME_FILE_NAME-'][-1]):
        window['-TAG2_FILE_NAME-'].update(values['-TAG2_FILE_NAME-'][:-1])
    if event == '-TAG2_FILE_NAME-' and values['-TAG2_FILE_NAME-'] and re.search(r'[0-9a-zA-Z\.]', values['-FRAME_FILE_NAME-'][-1]):
      window['-TAG2_FILE_NAME-'].update(background_color='#4A5C78')
    if event == '-TAG2_PERENT_MIN-' and values['-TAG2_PERENT_MIN-'] and values['-TAG2_PERENT_MIN-'][-1] not in ('0123456789.'):
      window['-TAG2_PERENT_MIN-'].update(values['-TAG2_PERENT_MIN-'][:-1])      
    if event == '-TAG2_PERENT_MIN-' and values['-TAG2_PERENT_MIN-'] and values['-TAG2_PERENT_MIN-'][-1] in ('0123456789.'):
      window['-TAG2_PERENT_MIN-'].update(background_color='#4A5C78')
    if event == '-TAG2_PERENT_MAX-' and values['-TAG2_PERENT_MAX-'] and values['-TAG2_PERENT_MAX-'][-1] not in ('0123456789.'):
      window['-TAG2_PERENT_MAX-'].update(values['-TAG2_PERENT_MAX-'][:-1])      
    if event == '-TAG2_PERENT_MAX-' and values['-TAG2_PERENT_MAX-'] and values['-TAG2_PERENT_MAX-'][-1] in ('0123456789.'):
      window['-TAG2_PERENT_MAX-'].update(background_color='#4A5C78')
    if event == '-TAG2_POSITION-' and values['-TAG2_POSITION-'] and values['-TAG2_POSITION-'][-1] not in ('0123456789,'):
      window['-TAG2_POSITION-'].update(values['-TAG2_POSITION-'][:-1])      
    if event == '-TAG2_POSITION-' and values['-TAG2_POSITION-'] and values['-TAG2_POSITION-'][-1] in ('0123456789,'):
      window['-TAG2_POSITION-'].update(background_color='#4A5C78')
    if event == '-TAG3_FILE_NAME-' and values['-TAG3_FILE_NAME-'] and not re.search(r'[0-9a-zA-Z\.]', values['-FRAME_FILE_NAME-'][-1]):
        window['-TAG3_FILE_NAME-'].update(values['-TAG3_FILE_NAME-'][:-1])
    if event == '-TAG3_FILE_NAME-' and values['-TAG3_FILE_NAME-'] and re.search(r'[0-9a-zA-Z\.]', values['-FRAME_FILE_NAME-'][-1]):
      window['-TAG3_FILE_NAME-'].update(background_color='#4A5C78')
    if event == '-TAG3_PERENT_MIN-' and values['-TAG3_PERENT_MIN-'] and values['-TAG3_PERENT_MIN-'][-1] not in ('0123456789.'):
      window['-TAG3_PERENT_MIN-'].update(values['-TAG3_PERENT_MIN-'][:-1])      
    if event == '-TAG3_PERENT_MIN-' and values['-TAG3_PERENT_MIN-'] and values['-TAG3_PERENT_MIN-'][-1] in ('0123456789.'):
      window['-TAG3_PERENT_MIN-'].update(background_color='#4A5C78')
    if event == '-TAG3_PERENT_MAX-' and values['-TAG3_PERENT_MAX-'] and values['-TAG3_PERENT_MAX-'][-1] not in ('0123456789.'):
      window['-TAG3_PERENT_MAX-'].update(values['-TAG3_PERENT_MAX-'][:-1])      
    if event == '-TAG3_PERENT_MAX-' and values['-TAG3_PERENT_MAX-'] and values['-TAG3_PERENT_MAX-'][-1] in ('0123456789.'):
      window['-TAG3_PERENT_MAX-'].update(background_color='#4A5C78')
    if event == '-TAG3_POSITION-' and values['-TAG3_POSITION-'] and values['-TAG3_POSITION-'][-1] not in ('0123456789,'):
      window['-TAG3_POSITION-'].update(values['-TAG3_POSITION-'][:-1])      
    if event == '-TAG3_POSITION-' and values['-TAG3_POSITION-'] and values['-TAG3_POSITION-'][-1] in ('0123456789,'):
      window['-TAG3_POSITION-'].update(background_color='#4A5C78')

    #ボタンを押した処理
    if event == "-START_BUTTON-":
      stop_flg = False
      window['-START_BUTTON-'].update('実行中', disabled=True, button_color=('white', 'grey'))
      window['-PROG_UNIT-'].UpdateBar(0)
      window['-PROG_GROUP-'].UpdateBar(0)

      #変数代入
      input_check = True
      input_err_text = ""

      if os.path.exists(values['-SOURCEDIRPATH-']):
        sorce_image_path = values['-SOURCEDIRPATH-']
      else:
        input_check = False
        input_err_text = "指定した合成素材フォルダが見つかりませんでした\n"
        window['-SOURCEDIRPATH-'].update(background_color='red')

      if os.path.exists(values['-RESULTDIRPATH-']):
        result_image_path = values['-RESULTDIRPATH-']
      else:
        input_check = False
        input_err_text = input_err_text + "指定した合成結果フォルダが見つかりませんでした\n"
        window['-RESULTDIRPATH-'].update(background_color='red')

      if values['-PIXCEL-'] != "":
        if int(values['-PIXCEL-']) >= 200 and int(values['-PIXCEL-']) <= 1000:
          pixcel = int(values['-PIXCEL-'])
        else:
          input_check = False
          input_err_text = input_err_text + "合成画像サイズに範囲外の値が入力されています\n"
          window['-PIXCEL-'].update(background_color='red')
      else:
        input_check = False
        input_err_text = input_err_text + "合成画像サイズに値が入力されていません\n"
        window['-PIXCEL-'].update(background_color='red')

      if values['-OUTPUT_MAX_NUM-'] != "":
        if int(values['-OUTPUT_MAX_NUM-']) >= 1 and int(values['-OUTPUT_MAX_NUM-']) <= 10:
          output_max_num = int(values['-OUTPUT_MAX_NUM-'])
        else:
          input_check = False
          input_err_text = input_err_text + "最大組み合わせ数に範囲外の値が入力されています\n"
          window['-OUTPUT_MAX_NUM-'].update(background_color='red')
      else:
        input_check = False
        input_err_text = input_err_text + "最大組み合わせ数に値が入力されていません\n"
        window['-OUTPUT_MAX_NUM-'].update(background_color='red')

      if values['-PERSON_FRAME_PERCENT-'] != "" and values['-PERSON_FRAME_PERCENT-'][0:1] != "." and values['-PERSON_FRAME_PERCENT-'][-1:] != ".":
        if float(values['-PERSON_FRAME_PERCENT-']) >= 0.0 and float(values['-PERSON_FRAME_PERCENT-']) <= 10.0:
          person_frame_percent = float(values['-PERSON_FRAME_PERCENT-'])*0.01
        else:
          input_check = False
          input_err_text = input_err_text + "人画像の余白に範囲外の値が入力されています\n"
          window['-PERSON_FRAME_PERCENT-'].update(background_color='red')
      else:
        input_check = False
        input_err_text = input_err_text + "人画像の余白に値が入力されていません\n"
        window['-PERSON_FRAME_PERCENT-'].update(background_color='red')

      if values['-PRODUCT_FRAME_PERCENT-'] != "" and values['-PRODUCT_FRAME_PERCENT-'][0:1] != "." and values['-PRODUCT_FRAME_PERCENT-'][-1:] != ".":
        if float(values['-PRODUCT_FRAME_PERCENT-']) >= 0.0 and float(values['-PRODUCT_FRAME_PERCENT-']) <= 10.0:
          product_frame_percent = float(values['-PRODUCT_FRAME_PERCENT-'])*0.01
        else:
          input_check = False
          input_err_text = input_err_text + "商品画像の余白に範囲外の値が入力されています\n"
          window['-PRODUCT_FRAME_PERCENT-'].update(background_color='red')
      else:
        input_check = False
        input_err_text = input_err_text + "商品画像の余白に値が入力されていません\n"
        window['-PRODUCT_FRAME_PERCENT-'].update(background_color='red')

      if values['-FRAME_FILE_NAME-'] != "":
        if values['-FRAME_FILE_NAME-'][-4:] == ".png":
          frame_file_name = values['-FRAME_FILE_NAME-']
        else:
          input_check = False
          input_err_text = input_err_text + "フレーム用の画像ファイルはPNG形式である必要があります\n"
          window['-FRAME_FILE_NAME-'].update(background_color='red')
      else:
        frame_file_name = values['-FRAME_FILE_NAME-']        
        window['-FRAME_FILE_NAME-'].update(background_color='red')

      if values['-TAG1_FILE_NAME-'] != "" and values['-TAG1_PERENT_MIN-'] != "" and values['-TAG1_PERENT_MAX-'] != "" and values['-TAG1_POSITION-'] != "":
        if values['-TAG1_FILE_NAME-'][-4:] == ".png":
          tag1_file_name = values['-TAG1_FILE_NAME-']
        else:
          input_check = False
          input_err_text = input_err_text + "挿入するファイル1はPNG形式である必要があります\n"
          window['-TAG1_FILE_NAME-'].update(background_color='red')
        if values['-TAG1_PERENT_MIN-'][0:1] != "." and values['-TAG1_PERENT_MIN-'][-1:] != "." and values['-TAG1_PERENT_MAX-'][0:1] != "." and values['-TAG1_PERENT_MAX-'][-1:] != ".":
          if float(values['-TAG1_PERENT_MIN-']) >= 1.0 and float(values['-TAG1_PERENT_MIN-']) <= 5.0 and float(values['-TAG1_PERENT_MAX-']) >= 1.0 and float(values['-TAG1_PERENT_MAX-']) <= 5.0 and float(values['-TAG1_PERENT_MAX-']) >= float(values['-TAG1_PERENT_MIN-']):
            tag1_percent_min = float(values['-TAG1_PERENT_MIN-'])*0.01
            tag1_percent_max = float(values['-TAG1_PERENT_MAX-'])*0.01
          else:
            input_check = False
            input_err_text = input_err_text + "挿入するファイル1の割合は1.0〜5.0の間で設定して下さい\n"
            window['-TAG1_PERENT_MIN-'].update(background_color='red')
            window['-TAG1_PERENT_MAX-'].update(background_color='red')
        else:
          input_check = False
          input_err_text = input_err_text + "挿入するファイル1の割合の数値が不正な値です\n"
          window['-TAG1_PERENT_MIN-'].update(background_color='red')
          window['-TAG1_PERENT_MAX-'].update(background_color='red')
        if not "" in [i for i in values['-TAG1_POSITION-'].split(",")]:
          tag1_position = [int(i) for i in values['-TAG1_POSITION-'].split(",")]
          if len([pos for pos in tag1_position if pos == 0 or pos > 25]) == 0:
            tag1_position = tag1_position
            if len(tag1_position) == len(set(tag1_position)):
              tag1_position = tag1_position
            else:
              input_check = False
              input_err_text = input_err_text + "挿入するファイル1の位置の候補に重複した数値が含まれています\n"
              window['-TAG1_POSITION-'].update(background_color='red')
          else:
            input_check = False
            input_err_text = input_err_text + "挿入するファイル1の位置の候補でに範囲外の数値が含まれています\n"
            window['-TAG1_POSITION-'].update(background_color='red')
        else:
          input_check = False
          input_err_text = input_err_text + "挿入するファイル1の位置の候補でカンマが不正な箇所に入っています\n"
          window['-TAG1_POSITION-'].update(background_color='red')
      else:
        if values['-TAG1_FILE_NAME-'] == "" and values['-TAG1_PERENT_MIN-'] == "" and values['-TAG1_PERENT_MAX-'] == "" and values['-TAG1_POSITION-'] == "":
          tag1_file_name = values['-TAG1_FILE_NAME-']
          tag1_percent_min = values['-TAG1_PERENT_MIN-']
          tag1_percent_max = values['-TAG1_PERENT_MAX-']
          tag1_position = values['-TAG1_POSITION-']
        else:
          input_check = False
          input_err_text = input_err_text + "挿入するファイル1に関する設定に未入力部分があります\n"
          if values['-TAG1_FILE_NAME-'] == "":
            window['-TAG1_FILE_NAME-'].update(background_color='red')
          if values['-TAG1_PERENT_MIN-'] == "":
            window['-TAG1_PERENT_MIN-'].update(background_color='red')
          if values['-TAG1_PERENT_MAX-'] == "":
            window['-TAG1_PERENT_MAX-'].update(background_color='red')
          if values['-TAG1_POSITION-'] == "":
            window['-TAG1_POSITION-'].update(background_color='red')

      if values['-TAG2_FILE_NAME-'] != "" and values['-TAG2_PERENT_MIN-'] != "" and values['-TAG2_PERENT_MAX-'] != "" and values['-TAG2_POSITION-'] != "":
        if values['-TAG2_FILE_NAME-'][-4:] == ".png":
          tag2_file_name = values['-TAG2_FILE_NAME-']
        else:
          input_check = False
          input_err_text = input_err_text + "挿入するファイル2はPNG形式である必要があります\n"
          window['-TAG2_FILE_NAME-'].update(background_color='red')
        if values['-TAG2_PERENT_MIN-'][0:1] != "." and values['-TAG2_PERENT_MIN-'][-1:] != "." and values['-TAG2_PERENT_MAX-'][0:1] != "." and values['-TAG2_PERENT_MAX-'][-1:] != ".":
          if float(values['-TAG2_PERENT_MIN-']) >= 1.0 and float(values['-TAG2_PERENT_MIN-']) <= 5.0 and float(values['-TAG2_PERENT_MAX-']) >= 1.0 and float(values['-TAG2_PERENT_MAX-']) <= 5.0 and float(values['-TAG2_PERENT_MAX-']) >= float(values['-TAG2_PERENT_MIN-']):
            tag2_percent_min = float(values['-TAG2_PERENT_MIN-'])*0.01
            tag2_percent_max = float(values['-TAG2_PERENT_MAX-'])*0.01
          else:
            input_check = False
            input_err_text = input_err_text + "挿入するファイル2の割合は1.0〜5.0の間で設定して下さい\n"
            window['-TAG2_PERENT_MIN-'].update(background_color='red')
            window['-TAG2_PERENT_MAX-'].update(background_color='red')
        else:
          input_check = False
          input_err_text = input_err_text + "挿入するファイル2の割合の数値が不正な値です\n"
          window['-TAG2_PERENT_MIN-'].update(background_color='red')
          window['-TAG2_PERENT_MAX-'].update(background_color='red')
        if not "" in [i for i in values['-TAG2_POSITION-'].split(",")]:
          tag2_position = [int(i) for i in values['-TAG2_POSITION-'].split(",")]
          if len([pos for pos in tag2_position if pos == 0 or pos > 25]) == 0:
            tag2_position = tag2_position
            if len(tag2_position) == len(set(tag2_position)):
              tag2_position = tag2_position
            else:
              input_check = False
              input_err_text = input_err_text + "挿入するファイル2の位置の候補に重複した数値が含まれています\n"
              window['-TAG2_POSITION-'].update(background_color='red')
          else:
            input_check = False
            input_err_text = input_err_text + "挿入するファイル2の位置の候補でに範囲外の数値が含まれています\n"
            window['-TAG2_POSITION-'].update(background_color='red')
        else:
          input_check = False
          input_err_text = input_err_text + "挿入するファイル2の位置の候補でカンマが不正な箇所に入っています\n"
          window['-TAG2_POSITION-'].update(background_color='red')
      else:
        if values['-TAG2_FILE_NAME-'] == "" and values['-TAG2_PERENT_MIN-'] == "" and values['-TAG2_PERENT_MAX-'] == "" and values['-TAG2_POSITION-'] == "":
          tag2_file_name = values['-TAG2_FILE_NAME-']
          tag2_percent_min = values['-TAG2_PERENT_MIN-']
          tag2_percent_max = values['-TAG2_PERENT_MAX-']
          tag2_position = values['-TAG2_POSITION-']
        else:
          input_check = False
          input_err_text = input_err_text + "挿入するファイル2に関する設定に未入力部分があります\n"
          if values['-TAG2_FILE_NAME-'] == "":
            window['-TAG2_FILE_NAME-'].update(background_color='red')
          if values['-TAG2_PERENT_MIN-'] == "":
            window['-TAG2_PERENT_MIN-'].update(background_color='red')
          if values['-TAG2_PERENT_MAX-'] == "":
            window['-TAG2_PERENT_MAX-'].update(background_color='red')
          if values['-TAG2_POSITION-'] == "":
            window['-TAG2_POSITION-'].update(background_color='red')

      if values['-TAG3_FILE_NAME-'] != "" and values['-TAG3_PERENT_MIN-'] != "" and values['-TAG3_PERENT_MAX-'] != "" and values['-TAG3_POSITION-'] != "":
        if values['-TAG3_FILE_NAME-'][-4:] == ".png":
          tag3_file_name = values['-TAG3_FILE_NAME-']
        else:
          input_check = False
          input_err_text = input_err_text + "挿入するファイル3はPNG形式である必要があります\n"
          window['-TAG3_FILE_NAME-'].update(background_color='red')
        if values['-TAG3_PERENT_MIN-'][0:1] != "." and values['-TAG3_PERENT_MIN-'][-1:] != "." and values['-TAG3_PERENT_MAX-'][0:1] != "." and values['-TAG3_PERENT_MAX-'][-1:] != ".":
          if float(values['-TAG3_PERENT_MIN-']) >= 1.0 and float(values['-TAG3_PERENT_MIN-']) <= 5.0 and float(values['-TAG3_PERENT_MAX-']) >= 1.0 and float(values['-TAG3_PERENT_MAX-']) <= 5.0 and float(values['-TAG3_PERENT_MAX-']) >= float(values['-TAG3_PERENT_MIN-']):
            tag3_percent_min = float(values['-TAG3_PERENT_MIN-'])*0.01
            tag3_percent_max = float(values['-TAG3_PERENT_MAX-'])*0.01
          else:
            input_check = False
            input_err_text = input_err_text + "挿入するファイル3の割合は1.0〜5.0の間で設定して下さい\n"
            window['-TAG3_PERENT_MIN-'].update(background_color='red')
            window['-TAG3_PERENT_MAX-'].update(background_color='red')
        else:
          input_check = False
          input_err_text = input_err_text + "挿入するファイル3の割合の数値が不正な値です\n"
          window['-TAG3_PERENT_MIN-'].update(background_color='red')
          window['-TAG3_PERENT_MAX-'].update(background_color='red')
        if not "" in [i for i in values['-TAG3_POSITION-'].split(",")]:
          tag3_position = [int(i) for i in values['-TAG3_POSITION-'].split(",")]
          if len([pos for pos in tag3_position if pos == 0 or pos > 25]) == 0:
            tag3_position = tag3_position
            if len(tag3_position) == len(set(tag3_position)):
              tag3_position = tag3_position
            else:
              input_check = False
              input_err_text = input_err_text + "挿入するファイル3の位置の候補に重複した数値が含まれています\n"
              window['-TAG3_POSITION-'].update(background_color='red')
          else:
            input_check = False
            input_err_text = input_err_text + "挿入するファイル3の位置の候補でに範囲外の数値が含まれています\n"
            window['-TAG3_POSITION-'].update(background_color='red')
        else:
          input_check = False
          input_err_text = input_err_text + "挿入するファイル3の位置の候補でカンマが不正な箇所に入っています\n"
          window['-TAG3_POSITION-'].update(background_color='red')
      else:
        if values['-TAG3_FILE_NAME-'] == "" and values['-TAG3_PERENT_MIN-'] == "" and values['-TAG3_PERENT_MAX-'] == "" and values['-TAG3_POSITION-'] == "":
          tag3_file_name = values['-TAG3_FILE_NAME-']
          tag3_percent_min = values['-TAG3_PERENT_MIN-']
          tag3_percent_max = values['-TAG3_PERENT_MAX-']
          tag3_position = values['-TAG3_POSITION-']
        else:
          input_check = False
          input_err_text = input_err_text + "挿入するファイル3に関する設定に未入力部分があります\n"
          if values['-TAG3_FILE_NAME-'] == "":
            window['-TAG3_FILE_NAME-'].update(background_color='red')
          if values['-TAG3_PERENT_MIN-'] == "":
            window['-TAG3_PERENT_MIN-'].update(background_color='red')
          if values['-TAG3_PERENT_MAX-'] == "":
            window['-TAG3_PERENT_MAX-'].update(background_color='red')
          if values['-TAG3_POSITION-'] == "":
            window['-TAG3_POSITION-'].update(background_color='red')

      #関数実行
      if input_check:

        window['-END_BUTTON-'].update(disabled=False)
        config['SECTION1']['pixcel'] = str(pixcel)
        config['SECTION1']['output_max_num'] = str(output_max_num)
        config['SECTION1']['person_frame_percent'] = str(person_frame_percent)
        config['SECTION1']['product_frame_percent'] = str(product_frame_percent)
        config['SECTION1']['frame_file_name'] = str(frame_file_name)
        config['SECTION1']['tag1_file_name'] = str(tag1_file_name)
        config['SECTION1']['tag1_percent_min'] = str(tag1_percent_min)
        config['SECTION1']['tag1_percent_max'] = str(tag1_percent_max)
        config['SECTION1']['tag1_position'] = '[' + ','.join(map(str, tag1_position)) + ']' if len(tag1_position) > 0 else tag1_position
        config['SECTION1']['tag2_file_name'] = str(tag2_file_name)
        config['SECTION1']['tag2_percent_min'] = str(tag2_percent_min)
        config['SECTION1']['tag2_percent_max'] = str(tag2_percent_max)
        config['SECTION1']['tag2_position'] = '[' + ','.join(map(str, tag2_position)) + ']' if len(tag2_position) > 0 else tag2_position
        config['SECTION1']['tag3_file_name'] = str(tag3_file_name)
        config['SECTION1']['tag3_percent_min'] = str(tag3_percent_min)
        config['SECTION1']['tag3_percent_max'] = str(tag3_percent_max)
        config['SECTION1']['tag3_position'] = '[' + ','.join(map(str, tag3_position)) + ']' if len(tag3_position) > 0 else tag3_position
        config['SECTION1']['sorce_image_path'] = str(sorce_image_path)
        config['SECTION1']['result_image_path'] = str(result_image_path)

        os.chdir(home_path)
        with open('./asset/conf/conf.ini', 'w') as f:
          config.write(f)

        threading.Thread(target=the_thread, args=(window,sorce_image_path,result_image_path,pixcel,output_max_num,person_frame_percent,product_frame_percent,frame_file_name,tag1_file_name,tag1_percent_min,tag1_percent_max,tag1_position,tag2_file_name,tag2_percent_min,tag2_percent_max,tag2_position,tag3_file_name,tag3_percent_min,tag3_percent_max,tag3_position), daemon=True).start()

      else:
        sg.popup(input_err_text,title = "設定に不備があります")
        window['-START_BUTTON-'].update('実行する', disabled=False, button_color=('white', 'green'))
        window['-TAB2-'].select()
    if event == "-END_BUTTON-":
      stop_flg = True
      window['-END_BUTTON-'].update('停止中', disabled=True, button_color=('white', 'grey'))

    #スレッドからのPUSH通知による処理
    if(event == THREAD_EVENT and values[THREAD_EVENT] == 'after'):
      display_after_image = get_display_data(display_image_path + '/' + 'display_after_image.png')
    if(event == THREAD_EVENT and values[THREAD_EVENT] == 'before'):
      display_before_image = get_display_data(display_image_path + '/' + 'display_before_image.png')
    if(event == THREAD_EVENT and values[THREAD_EVENT] == 'end'):
      window['-START_BUTTON-'].update('実行する', disabled=False, button_color=('white', 'green'))
      window['-END_BUTTON-'].update('停止する', disabled=True, button_color=('white', 'red'))
    if(event == THREAD_EVENT and values[THREAD_EVENT] == 'prog00'):
      window['-PROG_UNIT-'].UpdateBar(10)
    if(event == THREAD_EVENT and values[THREAD_EVENT] == 'prog01'):
      window['-PROG_UNIT-'].UpdateBar(20)
    if(event == THREAD_EVENT and values[THREAD_EVENT] == 'prog02'):
      window['-PROG_UNIT-'].UpdateBar(50)
    if(event == THREAD_EVENT and values[THREAD_EVENT] == 'prog03'):
      window['-PROG_UNIT-'].UpdateBar(60)
    if(event == THREAD_EVENT and values[THREAD_EVENT] == 'prog04'):
      window['-PROG_UNIT-'].UpdateBar(80)
    if(event == THREAD_EVENT and values[THREAD_EVENT] == 'prog05'):
      window['-PROG_UNIT-'].UpdateBar(100)
    if(event == THREAD_EVENT and type(values[THREAD_EVENT]) == int):
      window['-PROG_GROUP-'].UpdateBar(values[THREAD_EVENT])

    window['-DISPLAY_AFTER_IMAGE-'].update(data=display_after_image)
    window['-DISPLAY_BEFORE_IMAGE-'].update(data=display_before_image)
    

window.close()
