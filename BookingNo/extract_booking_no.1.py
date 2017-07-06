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
import os, glob, re
import csv


patterns = ["booking#", "booking"]
REGEX = r"[A-Z]{3,4}[\d]{3,12}"

# image_dir = "./tmp/cropped"
# image_dir = "/Users/thuong/Documents/SI/bookingNo-cropped"
image_dir = "./crop"


def draw_rectangle(img_file, crop_fname, x1, y1, x2, y2, color):
  im = Image.open(img_file)
  # dr = ImageDraw.Draw(im)
  # dr.rectangle(((x1,y1),(x2,y2)), fill=None, outline=color)
  # im.save(in_fname + ".png")
  im2 = im.crop((x1, y1, x2, y2))
  im2.save(os.path.join(crop_dir, crop_fname + "_crop.png"))


def extract_bookingno(text):
  blk_no = re.findall(REGEX, text) # match vs search (first find) vs findall()
  return blk_no


def file_count(dir, file_type):
  files_list = [f for f in os.listdir(dir) if f.endswith(file_type)]
  num_file = len(files_list)
  return num_file


def main():
  num_file = file_count(image_dir, '.png')
  i = 1
  num_bkl = 0
  for in_fname in glob.glob(os.path.join(image_dir, '*.png')):
    hocr_fname = in_fname.split("/")[-1]
    hocr_fname = hocr_fname.split(".")[0]
    print("(%s/%s) Tesseract on file %s" % (i, num_file, in_fname))
    text = pytesseract.image_to_string(Image.open(in_fname), config="-psm 6")
    book_no = extract_bookingno(text)
    i += 1
    print("Text .....: %s" % text)
    print("Found booking no.: %s" % book_no)
    print("Writing to file ...")
    print([book_no, in_fname])
    if len(book_no) > 0:
      num_bkl += 1
      print("Found %s/%s booking no." % (num_bkl, num_file))
      with open(r'./booking_no.csv', 'a', newline='\n') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        writer.writerow([book_no[0]])
    with open(r'./booking_no_map.csv', 'a', newline='\n') as csvfile:
      writer = csv.writer(csvfile, delimiter='\t')
      writer.writerow([book_no, str(in_fname)])


if __name__ == "__main__":
  main()