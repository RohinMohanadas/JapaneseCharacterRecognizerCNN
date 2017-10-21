import struct
from PIL import Image, ImageEnhance

filename = 'F:\KTH\DeepLearningProject\Code\dataset\ETL1\ETL1C_12'
skip = 100
with open(filename, 'rb') as f:
    f.seek(skip * 2052)
    s = f.read(2052)
    r = struct.unpack('>H2sH6BI4H4B4x2016s4x', s)
    iF = Image.frombytes('F', (64, 63), r[18], 'bit', 4)
    iP = iF.convert('P')
    fn = "{:1d}{:4d}{:2x}.png".format(r[0], r[2], r[3])
#    iP.save(fn, 'PNG', bits=4)
    enhancer = ImageEnhance.Brightness(iP)
    iE = enhancer.enhance(40)
    iE.save(fn, 'PNG')