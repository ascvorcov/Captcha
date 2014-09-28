import Image, ImageDraw, ImageFileIO, ImageFilter, ImageChops
import urllib
import collections
import os
import numpy as np
import pickle
import sys
from ffnet import loadnet, savenet, mlgraph, ffnet

Black = (0,0,0)
White = (255,255,255)
Processed = (255,0,0)
alphabet = "0123456789"

def loadimg(fn):
  im = Image.open(fn)
  im = im.convert("1").convert("RGB")

  im = ImageChops.invert(im)

  im = im.filter(ImageFilter.Kernel((3,3), [0,1,0,0,1,1,0,1,0]))
  #im = im.filter(ImageFilter.CONTOUR)
  #im = im.filter(ImageFilter.EDGE_ENHANCE)
  #im = im.filter(ImageFilter.EDGE_ENHANCE_MORE)
  
  im = im.crop(im.getbbox())
  #ts = [t/100.0 for t in range(101)]
  draw = ImageDraw.Draw(im)
  #draw.rectangle(getbounds(l), outline="green")
  #im = im.crop(im.getbbox())

  pts = []
  pbs = []
  
  step = im.size[0] / 10
  start = None
  for x in range(0, 10):
    crop = im.crop((x*step,0,(x+1)*step,100))
    box = crop.getbbox()
    if box != None:
      by = box[1]
      by2 = box[3]-1
      bx = -1
      bx2 = -1
      if start == None: start = (by,by2)

      for xx in range(0,step):
        if crop.getpixel((xx,by)) != Black and bx < 0:
          bx = xx
        if crop.getpixel((xx,by2)) != Black and bx2 < 0:
          bx2 = xx

      #if bx < 0: bx = 0
      #if bx2 < 0: bx2 = 0

      pts.append(((x*step)+bx,by))
      pbs.append(((x*step)+bx2,by2))

  #bz = make_bezier(pts)
  #draw.line(bz(ts), fill=128)
  pts.insert(0, (0,start[0]))
  pbs.insert(0, (0,start[1]))
  pts.append((im.size[0], by))
  pbs.append((im.size[0], by2))
  draw.line(pts, fill=255)
  draw.line(pbs, fill=255)
  del draw

  un = undeform(im)
  letters = full_check(un.copy())
  if letters == None:
    return []
    
  ret = []
  for lt in divide(paint_letters(letters)):
    ret.append(lt.resize((30,30)))
  return ret  

def divide(images):
  # divides input images into 6 ~equal pieces
  combinations = [ 
  (1,1,1,1,1,1), \
  (2,1,1,1,1), (1,2,1,1,1), (1,1,2,1,1), (1,1,1,2,1), (1,1,1,1,2), \
  (2,2,1,1), (2,1,2,1), (2,1,1,2), (1,1,2,2), (1,2,1,2), (1,2,2,1), (3,1,1,1), (1,3,1,1), (1,1,3,1), (1,1,1,3), \
  (1,2,3), (1,3,2), (2,1,3), (2,3,1), (3,1,2), (3,2,1), (4,1,1), (1,4,1), (1,1,4), (2,2,2), \
  (5,1), (1,5), (4,2), (2,4), (3,3), 
  (6,)]

  rects = [im.size for im in images]
  lst = [c for c in combinations if len(c) == len(rects)]
  tr = tuple(w * 1.0 / rects[0][0] for w,h in rects) #target rect, 1 tuple
  rr = [tuple(t * 1.0 / ct[0] for t in ct) for ct in lst] #referece rects, list of tuples
  distances = [sum((x1-x2)**2 for x1,x2 in zip(tr, r))**0.5 for r in rr] #sqrt((x1-x2)^2 + (y1-y2)^2...)
  div = lst[distances.index(min(distances))]
  ret = []
  for img,d in zip(images, div):
    for res in divide_img_pivot(img, d):
      ret.append(res)
  return ret

def divide_img(img, d):
  #divide image img into d parts
  if d == 1: return [img,]
  w,h = img.size
  step = w / (d * 1.0)
  ret = []
  for x in range(0, d):
    l = x * step
    bbox = (int(l), 0, int(l + step), h)
    ret.append(img.crop(bbox))

  return ret                            

def divide_img_pivot(img, d):
  #divide image img into d parts using local histogram minimum as division points
  if d == 1: return [img,]
  w,h = img.size
  step = w / (d * 1.0)
  hist = col_histogram(img)
  ret = []
  prev_pt = 0
  search_area = 12

  #cp = spacechart(img)
  #draw = ImageDraw.Draw(cp)
  for x in range(1, d):
    #choose current point
    cur_pt = prev_pt + step
    #find minimum in histogram subset in point neighborhood
    lt = int(cur_pt - search_area)
    rt = int(cur_pt + search_area)
    if rt > len(hist): 
      piece = hist[lt:]
    else: 
      piece = hist[lt:rt]
    offset = -search_area + piece.index(min(piece)) 
    cur_pt = int(cur_pt + offset)
    bbox = (prev_pt, 0, cur_pt, h)
    ret.append(img.crop(bbox))
    prev_pt = cur_pt
    #draw.line([(cur_pt,0),(cur_pt,10)], fill=(0,128,0))

  #del draw
  #cp.show()
  ret.append(img.crop((prev_pt, 0, w, h)))
  return ret                            

def paint_letters(letters):
  # accept list of list of two-tuples with letter pixel coordinates
  # paints each list as img
  ret = []
  for letter in letters:
    l,t,r,b = getbounds(letter)
    im = Image.new("RGB", (r-l+1,b-t+1))
    ret.append(im)
    for x,y in letter:
      x -= l
      y -= t
      im.putpixel((x,y), White)
  return ret

def col_histogram(im):
  # computes histogram for each image column as list (number of Black pixels)
  w,h = im.size
  ret = []
  for x in xrange(w):
    dh = sum(1 for y in xrange(h) if im.getpixel((x,y)) == Black)
    ret.append(dh)
  return ret

def spacechart(im):
  # draws white/black space chart for each img column
  w,h = im.size
  graph = Image.new("RGB", (w, h * 2))
  hist = col_histogram(im)
  for x,y in zip(xrange(w), hist):
    graph.putpixel((x,y), Processed)
  graph.paste(im, (0,h))
  return graph
  
def undeform(src):
  # for each scanline, find 2 red marks, forming bounding line
  # then deform image by bounding line, stretching imag
  dst = src.copy()
  for x in xrange(src.size[0]):
    foundtop = None
    foundbot = None
    height = src.size[1]
    for y in range(0, height):
      top = y
      bot = (height-1) - y
      if src.getpixel((x,top)) == Processed: foundtop = top
      if src.getpixel((x,bot)) == Processed: foundbot = bot
      if foundtop != None and foundbot != None:
        break

    dy = (foundbot-foundtop) / (1.0 * src.size[1])
    srcY = foundtop
    for y in xrange(src.size[1]):
      c = src.getpixel((x,srcY))
      if c == Processed: c = Black
      elif c != Black: c = White
      dst.putpixel((x,y), c)
      srcY += dy

  return dst    
  
def union_small(letters):
  # given list of shapes point coordinates, find smallest shapes 
  # and union with nearest large shape
  bounds = [getbounds(l) for l in letters]
  for b,l in zip(bounds, letters): 
    _,top,_,bottom = b
    if bottom - top < 30: # height < 30, too small
      intersecting = [cb for cb in bounds if cb != b and intersects(b,cb)]
    #todo  

def intersects(r1, r2):
    ax1,ay1,ax2,ay2 = r1
    bx1,by1,bx2,by2 = r2
    return ax1 < bx2 and ax2 > bx1 and ay1 < by2 and ay2 > by1

def full_check(img):
  #seeks letters in specified image, returns list of coordinates
  letters = []
  width, height = img.size
  pix = img.load()
  for x in range(0, width):
    for y in range(0, height):
      c = pix[x,y]
      if c == Black:
        fillimg(img, (x,y)) # ignore result, empty does not accumulate
      elif c != Processed:
        symbol = fillimg(img, (x,y))
        l,t,r,b = getbounds(symbol)
        if b - t < 30: return None # abort if floating disconnected chunk found (to do - join with nearest)
        letters.append(symbol)
      else: continue
  return letters

def trypush(pix, loc, w, h, stack, target):
  #tries to append specified location to stack, verifying stop conditions
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

def getbounds(letter):
  #returns bounding rectangle for specified list of points. (left,top,right,bottom)
  l,t,r,b = 10000,10000,0,0
  for x,y in letter:
    if x < l: l = x
    if x > r: r = x
    if y < t: t = y
    if y > b: b = y
  return (l,t,r,b)

def fillimg(img, loc):
  #takes point in image and floodfills it, returning as lst
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
    if c == target:
      lst.append(loc)
      x,y = loc
      trypush(pix, (x+1,y), w, h, pending, target)
      trypush(pix, (x-1,y), w, h, pending, target) 
      trypush(pix, (x,y+1), w, h, pending, target) 
      trypush(pix, (x,y-1), w, h, pending, target)

  return lst

def getbits(image):
  #returns bits of BW image as flat list of 1 and 0
  width, height = image.size
  pix = image.load()
  return [(1 if pix[x,y] == White else 0) for y in range(0, height) for x in range(0, width)]

def readsamples(path):
  #read samples from specified folder
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
      inputs = getbits(letter)
      pat.append([inputs, outputs])

  print "skipped:", skipped, ", processed:", processed

  return pat

def recognize(url):
  net = loadnet("ffnet.net")
  
  all_answers = []
  n_iter = 0
  while True:
    n_iter = n_iter + 1
    if n_iter > 15: break    
    print "iteration ", n_iter

    stream = ImageFileIO.ImageFileIO(urllib.urlopen(url))
    pat = loadimg(stream)
    if len(pat) != 6:
      continue

    answer = ""
    for letter in pat:
      arr = np.array(getbits(letter))
      answer += alphabet[net(arr).argmax()]
    all_answers.append(answer)
        
    result = []
    for x in range(0,6):
      what, count = collections.Counter([a[x] for a in all_answers]).most_common(1)[0]
      if count < 3: break # majority is 3 votes
      result.append(what)

    if len(result) == 6:
      print result
      break

  print all_answers
  Image.open(ImageFileIO.ImageFileIO(urllib.urlopen(url))).show()


def verify(data): 
  print "loading ffnet"
  net = loadnet("ffnet.net")
  
  success = 0;
  print "verifying %d samples" % len(data)

  for inp,out in data:
    arr = np.array(inp)
    result = net(arr)
    expected = alphabet[out.index(max(out))]
    recognized = alphabet[result.argmax()]
    if expected == recognized:
      success = success + 1
  print "%d of %d recognized, precision is %f" % (success, len(data), success / (len(data) * 1.0))
  
def train(data):
  #trains neural network based on passed training data.
  #training data is a list of [input,output] lists
   
  print "amount of training data:" + str(len(data))
  inputsize = 30 * 30
  outsize = 10
  nodes = 350 #((inputsize + outsize) * 2) / 3

  inp = [i for i,t in data]
  trg = [t for i,t in data]
    
  print "creating neural network, hidden nodes:" + str(nodes)
  conec = mlgraph((inputsize,nodes,outsize))
  
  print "initializing ffnet"
  net = ffnet(conec)
  
  #print "loading ffnet"
  #net = loadnet("ffnet.net")

  # print "assigning random weights"
  # net.randomweights()

  # Train process
  print "training network"
  net.train_tnc(inp, trg, messages=1,nproc=4)

  print "saving trained network"
  savenet(net, "ffnet.net")

  print "testing network"
  net.test(inp, trg, iprint = 1)
  
def make_bezier(xys):
    # xys should be a sequence of 2-tuples (Bezier control points)
    n = len(xys)
    combinations = pascal_row(n-1)
    def bezier(ts):
        result = []
        for t in ts:
            tpowers = (t**i for i in range(n))
            upowers = reversed([(1-t)**i for i in range(n)])
            coefs = [c*a*b for c, a, b in zip(combinations, tpowers, upowers)]
            result.append(
                tuple(sum([coef*p for coef, p in zip(coefs, ps)]) for ps in zip(*xys)))
        return result
    return bezier

def pascal_row(n):
    # This returns the nth row of Pascal's Triangle
    result = [1]
    x, numerator = 1, n
    for denominator in range(1, n//2+1):
        x *= numerator
        x /= denominator
        result.append(x)
        numerator -= 1
    if n&1 == 0:
        result.extend(reversed(result[:-1]))
    else:
        result.extend(reversed(result)) 
    return result

def geturl(url, p):
  f = urllib.urlopen(url)
  imgid = f.read()
  return url + p + imgid

if __name__ == '__main__':
  #samples = readsamples("./testset/")
  #samples = pickle.load( open( "save.p", "rb" ) )
  #train(samples)
  #verify(samples)
  recognize(geturl("http://...", "?"))