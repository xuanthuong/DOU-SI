from bs4 import BeautifulSoup
import collections
from PIL import Image, ImageFont, ImageDraw


def create_si_doc(hocr_file):
  with open(hocr_file) as fp:
    soup = BeautifulSoup(fp, "lxml")
  doc = []
  for line in soup.find_all('span', attrs={"class": 'ocr_line'}):
    tmp_line = []
    for text in line.find_all('span', attrs={"class": "ocrx_word"}):
      word = text.string
      if word and word.strip():
        tmp_arr = text.get('title').replace(";", "").split(" ")
        x1 = int(tmp_arr[1])
        y1 = int(tmp_arr[2])
        x2 = int(tmp_arr[3])
        y2 = int(tmp_arr[4])
        confdt = int(tmp_arr[6])
        tmp_line.append({
          "text": word,
          "x1": x1,
          "y1": y1,
          "x2": x2,
          "y2": y2,
          "confdt": confdt
        })
    doc.append(tmp_line)
  return doc


def cal_word_space(w1, w2):
  return abs(w2['x1'] - w1['x2'])


def find_word_space(doc):
  word_space = []
  for line in doc:
    for i in range(len(line) - 1):
      w1 = line[i]
      w2 = line[i+1]
      tmp_space = cal_word_space(w1, w2)
      word_space.append(tmp_space)
  # print(word_space)
  return collections.Counter(word_space).most_common()[0][0]


def find_line_space():
  pass


def combine_words_inline(doc, ws):
  tol = 20
  sentences = []
  for line in doc:
    first_word = True
    for i in range(len(line)):
      w1 = line[i]

      if first_word:
        if i == len(line) - 1:
          sentences.append({
            "content": w1['text'],
            "x1": w1['x1'],
            "y1": w1['y1'],
            "x2": w1['x2'],
            "y2": w1['y2']
          })
        else:
          sent = w1['text']
          x_left, y_left = w1['x1'], w1['y1']
          x_right, y_right = w1['x2'], w1['y2']
          first_word = False
      
      if i < len(line) - 1:
        w2 = line[i + 1]
        if abs(cal_word_space(w1, w2) - ws) < tol:
          sent = " ".join([sent, w2['text']])
          x_right, y_right = w2['x2'], w1['y2']
        else:
          sentences.append({
            "content": sent,
            "x1": x_left,
            "y1": y_left,
            "x2": x_right,
            "y2": y_right
          })
          first_word = True
      elif not first_word:
        sentences.append({
            "content": sent,
            "x1": x_left,
            "y1": y_left,
            "x2": w1['x2'],
            "y2": w1['y2']
          })
  return sentences


def draw_rectangle(img_file, sentences, color = "red"):
  im = Image.open(img_file)
  dr = ImageDraw.Draw(im)
  for sent in sentences:
    if sent['content'] and sent['content'].strip():
      dr.rectangle(((sent['x1'],sent['y1']),(sent['x2'],sent['y2'])), fill=None, outline=color)
  im.save("./out_rec" + ".png")


def main():
  doc = create_si_doc("./100.hocr")
  word_space = find_word_space(doc)
  # print("word space: %s" % word_space)
  sentences = combine_words_inline(doc, word_space)
  draw_rectangle("./100.png", sentences)


if __name__ == "__main__":
  main()


