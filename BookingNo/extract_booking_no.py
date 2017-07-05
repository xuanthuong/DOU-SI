#  -*- coding: utf-8 -*-
"""
Extract Booking number from SI documents
Methodology: OCR 
reference: 

Date: Jul 04, 2017
@author: Thuong Tran
@Library: Tesseract-OCR - pytesseract
https://stackoverflow.com/questions/29141840/script-that-converts-html-tables-to-csv-preferably-python
"""


from PIL import Image, ImageFont, ImageDraw
import pytesseract
from bs4 import BeautifulSoup
import os, glob

patterns = ["booking#", "booking"]

image_dir = "/Users/thuong/Documents/SI/images/SI"
crop_dir = "/Users/thuong/Documents/SI/bookingNo-cropped"
hocr_dir = "/Users/thuong/Documents/SI/hocr"
# in_image = "./sampleSI/1.png"
# in_fname = in_image.split("/")[-1]
# in_fname = in_fname.split(".")[0]

LEFT_PADDING = 10
TOP_PADDING = 10
RIGHT_PADDING = 3
BOTTOM_PADDING = 3


def draw_rectangle(img_file, crop_fname, x1, y1, x2, y2, color):
  im = Image.open(img_file)
  # dr = ImageDraw.Draw(im)
  # dr.rectangle(((x1,y1),(x2,y2)), fill=None, outline=color)
  # im.save(in_fname + ".png")
  im2 = im.crop((x1, y1, x2, y2))
  im2.save(os.path.join(crop_dir, crop_fname + "_crop.png"))


def main():
  for in_fname in glob.glob(os.path.join(image_dir, '*.png')):
    hocr_fname = in_fname.split("/")[-1]
    hocr_fname = hocr_fname.split(".")[0]
    print("Tesseract on file %s" % in_fname)
    pytesseract.pytesseract.run_tesseract(in_fname, os.path.join(hocr_dir, hocr_fname), 
                                          boxes=False, config="-psm 3 hocr")
    hocr_file = os.path.join(hocr_dir, hocr_fname + ".hocr")

    with open(hocr_file) as fp:
      soup = BeautifulSoup(fp, "lxml")
    print("Starting find booking no...")
    for text in soup.find_all('span', {'class': 'ocrx_word'}):
      wrd = text.string
      wrd = wrd.lower()
      if wrd == patterns[0] or wrd == patterns[1]:
        tmp_arr = text.get('title').replace(";", "").split(" ")
        x1 = int(tmp_arr[1])
        y1 = int(tmp_arr[2])
        x2 = int(tmp_arr[3])
        y2 = int(tmp_arr[4])
        draw_rectangle(in_fname, hocr_fname, 
                        x1 - LEFT_PADDING, 
                        y1 - TOP_PADDING,
                        x2 + RIGHT_PADDING * (x2 - x1),
                        y2 + BOTTOM_PADDING * (y2 - y1), "green")
        print("Found keyword: %s" % wrd)
        break


if __name__ == "__main__":
  main()