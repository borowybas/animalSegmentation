import os
from PIL import Image

DATA_DIR = './dataset/'
images_dir = os.path.join(DATA_DIR, 'Images')


ids = os.listdir(images_dir)
for file in ids:
    path = os.path.join(images_dir, file)

    img = Image.open(path)
    newpath = os.path.splitext(path)[0] + '.png'
    img.save(newpath)
    if not path.endswith('.png'):
        os.remove(path)
