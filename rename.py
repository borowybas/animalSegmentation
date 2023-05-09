import os
DATA_DIR = './dataset/'

images_dir = os.path.join(DATA_DIR, 'Images')
masks_dir = os.path.join(DATA_DIR, 'Masks')
datadir = masks_dir

polish = ['ą', 'ć', 'ę', 'ł', 'ń', 'ó', 'ś', 'ż', 'ź', 'ę', 'ż']
english = ['a', 'c', 'e', 'l', 'n', 'o', 's', 'z', 'z', 'e', 'z']

ids = os.listdir(datadir)
for path in ids:
    old_path = path
    for i, sign in enumerate(polish):
        if sign in path:
            path = path.replace(sign, english[i])
        if sign.upper() in path:
            path = path.replace(sign.upper(), english[i].upper())
    print(old_path, '/n', path)
    old_path = os.path.join(datadir, old_path)
    path = os.path.join(datadir, path)
    if old_path != path:
        os.rename(old_path, path)
