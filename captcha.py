import Image, ImageDraw, ImageFileIO
import urllib
import os
import numpy as np
import neurolab as nl
import pickle
from ffnet import loadnet

Black = (0,0,0)
White = (255,255,255)
Processed = (255,0,0)
alphabet = "ACDEFGHJKLNPQRTUVXYZ2346789"

#loads image and recognizes letters on it
def loadimg(filename):
  imgfile = None
  if filename[:4] == "http":
    stream = ImageFileIO.ImageFileIO(urllib.urlopen(filename))
    imgfile = Image.open(stream)
  else:
    imgfile = Image.open(filename)

  imgfile = imgfile.convert("1").convert("RGB")
  letters = full_check(imgfile)
  letters.sort(key=lambda l: l[0][0])
  return letters

#draws specified letter in 30x30 BW canvas and return it
def drawletter(letter):
  canvas = Image.new("1", (30,30), 0)
  pix = canvas.load()
  for loc in center(letter):
    pix[loc] = 1
  return canvas

#returns bits of BW image as flat list of 1 and 0
def getbits(image):
  width, height = image.size
  pix = image.load()
  return [pix[x,y] for y in range(0, height) for x in range(0, width)]


#creates test image showing which part was recognized
def testimg(filename):
  letters = loadimg(filename)
 
  print str(len(letters)) + " letters recognized"
  copy = Image.open(filename)
  copy = copy.convert("1").convert("RGB")
  pix = copy.load()
  for l in letters:
    draw = ImageDraw.Draw(copy)
    draw.rectangle(getbounds(l), outline="green")
    del draw
    for coord in l:
      x,y = coord
      pix[x,y] = Processed
  copy.save("q.bmp")


#seeks letters in specified image, returns list of coordinates
def full_check(img):
  letters = []
  width, height = img.size
  pix = img.load()
  for x in range(0, width):
    for y in range(0, height):
      c = pix[x,y]
      if c == White:
        fillimg(img, (x,y)) # ignore result, whitespace does not accumulate
      elif c == Black:
        symbol = fillimg(img, (x,y))
        if len(symbol) > 20: # is big enough?
          letters.append(symbol)
      elif c == Processed:
        continue
      else: 
        raise ValueError("Only BW images supported:" + str(c))
  return letters

#tries to append specified location to stack, verifying stop conditions
def trypush(pix, loc, w, h, stack, target):
  x, y = loc
  if x < 0 or x >= w:
    return
  if y < 0 or y >= h:
    return
  if pix[x,y] != target:
    return
  if loc in stack:
    return
  stack.add(loc)

#returns bounding rectangle for specified list of points. (left,top,right,bottom)
def getbounds(letter):
  l,t,r,b = 10000,10000,0,0
  for x,y in letter:
    if x < l: l = x
    if x > r: r = x
    if y < t: t = y
    if y > b: b = y
  return (l,t,r,b)

#centers specified letter in 30x30 bounding rect, returns new letter
def center(letter):
  result = []
  l,t,r,b = getbounds(letter)
  for x,y in letter:
    x = x - l + 15 - (r - l)/2
    y = y - t + 15 - (b - t)/2
    if x < 0 or y < 0 or x >= 30 or y >= 30:
      continue
    result.append((x,y))
  return result

#takes point in image and floodfills it, returning as lst
def fillimg(img, loc):

  lst = []
  w, h = img.size
  pix = img.load()
  target = pix[loc]

  if target == Processed:
    return lst

  pending = set([loc])

  while len(pending) > 0:
    loc = pending.pop()
    c = pix[loc]
    pix[loc] = Processed
    if c == Processed:
      raise ValueError("Encountered processed pixel while traversing image.")
    if c != Black and c != White:
      raise ValueError("Only BW images supported:" + str(c))
    if c == target:
      lst.append(loc)
      x,y = loc
      trypush(pix, (x+1,y), w, h, pending, target)
      trypush(pix, (x-1,y), w, h, pending, target) 
      trypush(pix, (x,y+1), w, h, pending, target) 
      trypush(pix, (x,y-1), w, h, pending, target)

  return lst

#read samples from specified folder
def readsamples(path):

    pat = []
    # each symbol is fit into 30x30 image
    inputsize = 30 * 30

    processed = 0
    skipped = 0
    for fn in os.listdir(path):
      print "processing " + fn
      pics = loadimg(path + fn)
      code = fn[:-4]
      if len(pics) != len(code):
        print "skipping, recognized symbols:" + str(len(pics))
        skipped = skipped + 1
        continue;
      processed = processed + 1

      for char, letter in zip(code, pics):
        outputs = [0]*len(alphabet)
        outputs[alphabet.index(char)] = 1
        inputs = getbits(drawletter(letter))
        pat.append([inputs, outputs])

    return pat

#try to recognize single 30x30 image, return letter from alphabet
def recognizeletter(net, inputbits):
  arr = np.array(inputbits)
  result = net(arr)
  return alphabet[result.argmax()]

#trains neural network based on passed training data.
#training data is a list of [input,output] lists
def train(data):
   
    print "amount of training data:" + str(len(data))
    inputsize = 30 * 30
    outsize = 27
    nodes = ((inputsize + outsize) * 2) / 3
    print "creating neural network, hidden nodes:" + str(nodes)

    inp = np.array([i[0] for i in data]).reshape(len(data), inputsize)
    out = np.array([i[1] for i in data]).reshape(len(data), outsize)

    #input = (samples_count, input_neurons)
    #output = (samples_count, output_neurons)
 
    # Create network with 900 inputs, ~300 neurons in hidden layer and 27 in output layer
    net = nl.net.newff([[0.0, 1.0]] * inputsize, [nodes, outsize])
    # Train process
    err = net.train(inp, out, epochs=500, show=1, goal=0.02)
    print err
    net.save('_captcha.net')

def savedata(data):
  f = open("_captcha.p", "wb")
  pickle.dump(data, f)
  f.close()

def loaddata():
  f = open("_captcha.p", "rb")
  d = pickle.load(f)
  f.close()
  return d

def saveconvert(data):
  f = open("_captcha.txt", "w")
  for d in data:
    f.write(str(d[0])[1:-1])
    f.write(", ")
    f.write(str(d[1])[1:-1])
    f.write("\n")
  f.close()

def recognize(imgpath):
  net = loadnet("captcha.net")
  pics = loadimg(imgpath)
  if len(pics) != 5:
    print "skipping, recognized symbols:" + str(len(pics))
    return;

  res = ""
  for letter in pics:
    inputbits = getbits(drawletter(letter))
    res = res + recognizeletter(net, inputbits)
  print res

def host():
  import xmlrpclib
  from SimpleXMLRPCServer import SimpleXMLRPCServer
  server = SimpleXMLRPCServer(("localhost", 8000))
  print "Listening on port 8000..."
  server.register_function(recognize, "recognize")
  server.serve_forever()


if __name__ == '__main__':
  host()
  #testimg("./cap/images/27JR6.bmp")
  #s = readsamples("./cap/images/")
  #saveconvert(loaddata())
  #recognize("https://.../MyCaptchaImage.aspx?guid=3ef4e21d-8629-4e38-9fc4-f0c992004ac3")
  #train(loaddata())
