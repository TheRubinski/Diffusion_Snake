import numpy as np
from PIL import Image, ImageDraw

#polygon = [(x1,y1),(x2,y2),...] or [x1,y1,x2,y2,...]
# width = ?
# height = ?

polygon=np.array([(10,10),(90,10),(90,80),(10,60)])
img = Image.new('L', (100, 100), 0)
ImageDraw.Draw(img).polygon(polygon, outline=1, fill=2)
mask = np.array(img)

from matplotlib import pyplot as plt
plt.imshow(mask)
plt.show()