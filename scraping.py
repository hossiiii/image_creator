from selenium import webdriver # さっきpip install seleniumで入れたseleniumのwebdriverというやつを使う
import datetime
import os,csv
import requests
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import NoSuchElementException
from selenium.common.exceptions import InvalidArgumentException
import tkinter, tkinter.filedialog, tkinter.messagebox
import re
import time
import sys
import struct
import urllib.request

def parse_jpeg(res):
    while not res.closed:
        (marker, size) = struct.unpack('>2sH', res.read(4))
        if marker == b'\xff\xc0':
            (_,height,width,_) = struct.unpack('>chh10s', res.read(size-2))
            return (width,height)
        else:
            res.read(size-2)

def parse_png(res):
    (_,width,height) = struct.unpack(">14sII", res.read(22))
    return (width, height)

def parse_gif(res):
    (_,width,height) = struct.unpack("<4sHH", res.read(8))
    return (width, height)

def get_image_size(url):
    res = urllib.request.urlopen(url)
    size = (-1,-1)
    if res.status == 200:
        signature = res.read(2)
        if signature == b'\xff\xd8': #jpg
            size = parse_jpeg(res)
        elif signature == b'\x89\x50': #png
            size = parse_png(res)
        elif signature == b'\x47\x49': #gif
            size = parse_gif(res)
    res.close()
    return size

options = Options()
UA = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36'
options.add_argument('--user-agent=' + UA)
options.add_experimental_option("excludeSwitches", ["enable-automation"])
#options.add_argument('--headless')
driver = webdriver.Chrome("/Users/hossiiii0117/python/essam/chromedriver",chrome_options=options) # さっきDLしたchromedriver.exeを使う
# driver = webdriver.Chrome("/Users/hossiiii0117/python/essam/chromedriver") # さっきDLしたchromedriver.exeを使う
# driver.implicitly_wait(10) # seconds
driver.get("https://www.hermes.com/jp/ja/product/%E3%82%AD%E3%83%A3%E3%82%B9%E3%82%B1%E3%83%83%E3%83%88-%E3%80%8A%E3%83%A9%E3%82%A4%E3%83%AA%E3%83%BC%E3%80%8B-H221044Nv9158/")
# driver.get("https://www.gucci.com/jp/ja/pr/women/handbags/shoulder-bags-for-women/gucci-horsebit-1955-shoulder-bag-p-602204UQKAG1058")

time.sleep(10)

imgs = driver.find_elements_by_tag_name('img')
for img in imgs:
	imgurl = img.get_attribute('src')
	print(imgurl)
	if(imgurl != None):
		print(get_image_size(imgurl))
		with open("out.png", 'wb') as f:
			f.write(img.screenshot_as_png)

driver.close()
driver.quit()

driver.get("https://www.hermes.com/jp/ja/product/%E3%82%AD%E3%83%A3%E3%82%B9%E3%82%B1%E3%83%83%E3%83%88-%E3%80%8A%E3%83%A9%E3%82%A4%E3%83%AA%E3%83%BC%E3%80%8B-H221044Nv9158/")
# driver.get("https://www.gucci.com/jp/ja/pr/women/handbags/shoulder-bags-for-women/gucci-horsebit-1955-shoulder-bag-p-602204UQKAG1058")

time.sleep(10)

imgs = driver.find_elements_by_tag_name('img')
for img in imgs:
	imgurl = img.get_attribute('src')
	print(imgurl)
	if(imgurl != None):
		print(get_image_size(imgurl))
		with open("out.png", 'wb') as f:
			f.write(img.screenshot_as_png)
