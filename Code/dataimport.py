

import struct
import os
import platform
from PIL import Image, ImageEnhance
import traceback
import logging
import numpy as np

# Meta Data should have the pattern:
# [Path, categories, sheets, rec_len, JIS X Notation Index, Picture Index]
ETL1C_META_MAC=    (('/ETL1/ETL1C_01',8,1445 ,2052, 3, 18),
                    ('/ETL1/ETL1C_02',8,1445 ,2052, 3, 18),
                    ('/ETL1/ETL1C_03',8,1445 ,2052, 3, 18),
                    ('/ETL1/ETL1C_04',8,1445 ,2052, 3, 18),
                    ('/ETL1/ETL1C_05',8,1445 ,2052, 3, 18),
                    ('/ETL1/ETL1C_06',8,1445 ,2052, 3, 18),
                    ('/ETL1/ETL1C_07',8,1411 ,2052, 3, 18),
                    ('/ETL1/ETL1C_08',8,1411 ,2052, 3, 18),
                    ('/ETL1/ETL1C_09',8,1411 ,2052, 3, 18),
                    ('/ETL1/ETL1C_10',8,1411 ,2052, 3, 18),
                    ('/ETL1/ETL1C_11',8,1411 ,2052, 3, 18),
                    ('/ETL1/ETL1C_12',8,1411 ,2052, 3, 18),
                    ('/ETL1/ETL1C_13',3,1411 ,2052, 3, 18))

ETL1C_META_WIN =    (('\\ETL1\\ETL1C_01',8,1445 ,2052, 3, 18),
                    ('\\ETL1\\ETL1C_02',8,1445 ,2052, 3, 18),
                    ('\\ETL1\\ETL1C_03',8,1445 ,2052, 3, 18),
                    ('\\ETL1\\ETL1C_04',8,1445 ,2052, 3, 18),
                    ('\\ETL1\\ETL1C_05',8,1445 ,2052, 3, 18),
                    ('\\ETL1\\ETL1C_06',8,1445 ,2052, 3, 18),
                    ('\\ETL1\\ETL1C_07',8,1411 ,2052, 3, 18),
                    ('\\ETL1\\ETL1C_08',8,1411 ,2052, 3, 18),
                    ('\\ETL1\\ETL1C_09',8,1411 ,2052, 3, 18),
                    ('\\ETL1\\ETL1C_10',8,1411 ,2052, 3, 18),
                    ('\\ETL1\\ETL1C_11',8,1411 ,2052, 3, 18),
                    ('\\ETL1\\ETL1C_12',8,1411 ,2052, 3, 18),
                    ('\\ETL1\\ETL1C_13',3,1411 ,2052, 3, 18))

ETL8B_META_MAC =     (('/ETL8B/ETL8B2C1',320,160 ,512, 1, 3),
                    ('/ETL8B/ETL8B2C2',320,160 ,512, 1, 3),
                    ('/ETL8B/ETL8B2C3',316,160 ,512, 1, 3))
ETL8B_META_WIN =     (('\\ETL8B\\ETL8B2C1',320,160 ,512, 1, 3),
                    ('\\ETL8B\\ETL8B2C2',320,160 ,512, 1, 3),
                    ('\\ETL8B\\ETL8B2C3',316,160 ,512, 1, 3))

FORBIDDEN_KATAKANA_SHEETS = list(range(1191,1197)) + list(range(1234,1243)) + [2011, 2911]

# ARGS:
#   dataset :   Defines which set of characters we want
def import_data(dataset='ALL', vectorize = False, image_brightness = 35, num_image = None):
    factors = []
    labels = []
    if platform.system() in ["Darwin","Linux"]:
        filename = os.path.dirname(os.path.dirname((os.path.abspath(__file__)))) + '/dataset'
    else:
        filename = os.path.dirname(os.path.dirname((os.path.abspath(__file__)))) + '\\dataset'
    if dataset in ['KATAKANA', 'ALL']:
        for row in range(6,13):
            if platform.system() in ["Darwin","Linux"]:
                part = ETL1C_META_MAC[row]
            else:
                part = ETL1C_META_WIN[row]
            if num_image:
                num_sheets = num_image
            else:
                num_sheets = part[2]

            f_part, l_part = import_dataset_part(
                                    path = filename + part[0],
                                    num_of_categories = part[1],
                                    sheets = num_sheets,
                                    record_len = part[3],
                                    database = 'ETL1C',
                                    vectorize = vectorize,
                                    image_brightness = image_brightness)
            factors += f_part
            labels += l_part
        print("Imported KATAKANA")
    if dataset in ['KANJI', 'ALL']:
        if platform.system() in ["Darwin","Linux"]:
            dataset_kanji = ETL8B_META_MAC
        else:
            dataset_kanji = ETL8B_META_WIN
        separate_hiragana = True
        for part in dataset_kanji:
            if num_image:
                num_sheets = num_image
            else:
                num_sheets = part[2]
            if separate_hiragana:
                lower_edge = 75
                separate_hiragana = False
            else:
                lower_edge = 0
            f_part, l_part = import_dataset_part(
                                    path = filename + part[0],
                                    num_of_categories = part[1],
                                    sheets = num_sheets,
                                    record_len = part[3],
                                    database = 'ETL8B2',
                                    vectorize = vectorize,
                                    image_brightness = image_brightness,
                                    lower_edge_of_categories = lower_edge)
            factors += f_part
            labels += l_part
        print("Imported KANJI")
    if dataset in ['HIRAGANA', 'ALL']:
        if platform.system() in ["Darwin","Linux"]:
            dataset_kanji = ETL8B_META_MAC[0]
        else:
            dataset_kanji = ETL8B_META_WIN[0]
        part = dataset_kanji
        if num_image:
            num_sheets = num_image
        else:
            num_sheets = part[2]
        # num_sheets = 37
        f_part, l_part = import_dataset_part(
                                path = filename + part[0],
                                num_of_categories = 75,
                                sheets = num_sheets,
                                record_len = part[3],
                                database = 'ETL8B2',
                                vectorize = vectorize,
                                image_brightness = image_brightness)
        factors += f_part
        labels += l_part
        print("Imported HIRAGANA")
    return factors, labels

# ARGS:
#   f           :   The file which we are reading from
#   record_len  :   The records length in bytes
#   num_char    :   Number of copies of each char to parse
#   size        :   Tuple with Width and Hight of pictures
#   database    :   Defines which database data is read from
def save_data(f, record_len, num_char, size, database, vectorize, image_brightness):
    (W, H) = size
    a = 0
    iI = None
    for i in range(0,num_char):
        new_img = Image.new('1', (W, 64))

        if database == 'ETL8B2':
            s = f.read(512)
            r = struct.unpack('>2H4s504s', s)
            iE = Image.frombytes('1', (W, H), r[3], 'raw')
            # print(r[0:3], hex(r[1]))
            new_img.paste(iE, (0,0))
            iI = Image.eval(new_img, lambda x: not x)
            if platform.system() in ["Darwin","Linux"]:
                fn = "TestPics/{}{}.png".format( r[1],r[0] )
            else:
                fn = "F:\KTH\DeepLearningProject\Code\TestPics\{}.png".format( r[1])
        # elif database == 'ETL1C'
        else:
            s = f.read(record_len)
            r = struct.unpack('>H2sH6BI4H4B4x2016s4x', s)
            iF = Image.frombytes('F', (W, H), r[18], 'bit', 4)
            iP = iF.convert('P')

            if platform.system() in ["Darwin","Linux"]:
                fn = "TestPics/{:4d}{:2x}.png".format( r[2], r[3])
                fn2 = "TestPics/{:4d}{:2x}newim.png".format( r[2], r[3])
                fn3 = "TestPics/{:4d}{:2x}iI.png".format( r[2], r[3])
            else:
                fn = "F:\KTH\DeepLearningProject\Code\TestPics\{:4d}{:2x}.png".format( r[2], r[3])
                fn2 = "F:\KTH\DeepLearningProject\Code\TestPics\{:4d}{:2x}newim.png".format( r[2], r[3])
                fn3 = "F:\KTH\DeepLearningProject\Code\TestPics\{:4d}{:2x}iI.png".format( r[2], r[3])
            #print('r0='+str(r[0])+', r1='+str(r[1]) +', r2='+str(r[2]) + ', r3='+str(r[3]))
            # Adjust image brightness.
            # This class can be used to control the brightness of an image.
            # An enhancement factor of 0.0 gives a black image.
            # A factor of 1.0 gives the original image.
            #------------------------------------------
            enhancer = ImageEnhance.Brightness(iP)
            #-------------------------------------
            # Factor 1.0 always returns a copy of the original image,
            # lower factors mean less color (brightness, contrast, etc),
            # and higher values more. There are no restrictions on this value.
            #--------------------------

            # iE = enhancer.enhance(35)

            iE = enhancer.enhance(image_brightness)

            #--------------------------
            size_add = 12
            iE = iE.resize((W + size_add, H + size_add))
            iE = iE.crop((size_add / 2,
                        size_add / 2,
                        W + size_add / 2,
                        H + size_add / 2))
            r = r + (iE,)
            new_img.paste(r[-1],(0,0))
            iI = Image.eval(new_img, lambda x: not x)
        if vectorize:
            outData = np.asarray(iI.getdata()).reshape(
                            64 * 64)
        else:
            outData = iI
        if database == 'ETL1C' and r[2] in FORBIDDEN_KATAKANA_SHEETS:
            a += 1
        else:
            # iE.save(fn, 'PNG')
            new_img.save(fn, 'PNG')
            # iI.save(fn3, 'PNG')
    return [1],[1]

# ARGS:
#   f           :   The file which we are reading from
#   record_len  :   The records length in bytes
#   num_char    :   Number of copies of each char to parse
#   size        :   Tuple with Width and Hight of pictures
#   database    :   Defines which database data is read from
def parse_data(f, record_len, num_char, size, database, vectorize, image_brightness):
    a=0
    feature_part = []
    labels_part = []
    (W, H) = size
    Hnew = 64
    # resize to 64 * 64
    new_img = Image.new('1', (W, Hnew))
    for i in range(0,num_char-11):
        if database == 'ETL8B2':
            s = f.read(512)
            try:
                r = struct.unpack('>2H4s504s', s)
                iE = Image.frombytes('1', (W, H), r[3], 'raw')
                # print(r[0:3], hex(r[1]))
                new_img.paste(iE, (0,0))
                iI = Image.eval(new_img, lambda x: not x)
                labels_part.append(r[1])
                feature_part += [list(iI.getdata())]
            except Exception as e:
                print('Parse image Exception at ' + str(i))
                logging.error(traceback.format_exc())
        elif database == 'ETL1C':
            s = f.read(record_len)
            try:
                r = struct.unpack('>H2sH6BI4H4B4x2016s4x', s)
                iF = Image.frombytes('F', (W, H), r[18], 'bit', 4)
                iP = iF.convert('P')
                # Adjust image brightness.
                # This class can be used to control the brightness of an image.
                # An enhancement factor of 0.0 gives a black image.
                # A factor of 1.0 gives the original image.
                #------------------------------------------
                enhancer = ImageEnhance.Brightness(iP)
                #-------------------------------------
                # Factor 1.0 always returns a copy of the original image,
                # lower factors mean less color (brightness, contrast, etc),
                # and higher values more. There are no restrictions on this value.
                #--------------------------
                iE = enhancer.enhance(35)
                #--------------------------
                # expand to crop and fit to 64*64
                size_add = 12
                iE = iE.resize((W + size_add, H + size_add))
                iE = iE.crop((size_add / 2,
                            size_add / 2,
                            W + size_add / 2,
                            H + size_add / 2))

                # r = r + (iE,)
                # new_img.paste(r[-1],(0,0))
                # iI = Image.eval(new_img, lambda x: not x)

                # labels_part += [r[3]]
                # feature_part += [list(iI.getdata())]

                r = r + (iE,)
                new_img.paste(r[-1],(0,0))
                iI = Image.eval(new_img, lambda x: not x)
                if vectorize:
                    outData = np.asarray(iI.getdata()).reshape(
                                    64 * 64)
                else:
                    outData = list(iI.getdata())
                if database == 'ETL1C' and r[2] in FORBIDDEN_KATAKANA_SHEETS:
                    a+=0
                else:
                    labels_part += [r[3]]
                    feature_part += [outData]

            except Exception as e:
                print('Parse Exception at ' + str(i))
                print(record_len)
                print(s)
                logging.error(traceback.format_exc())
    if len(feature_part) != len(labels_part):
        print('Features and labels differ here: ')
        print(error_data)

    return feature_part, labels_part

# ARGS:
#   sheets      :   The number of writers
#   record_len  :   The records length in bytes
#   W, H        :   Width and Hight of pictures
#   categories  :   Defined as 8 for ETL1C-01~ETL1C-12, and 3 in ETL1C-13
def import_dataset_part(path, num_of_categories,
                            sheets, record_len, database, vectorize = False, image_brightness = 20, size = (64,63), lower_edge_of_categories=0):
    #filename = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/dataset' + dataset + part
    filename = path
    categories = range(lower_edge_of_categories,num_of_categories)
    feature = []
    labels = []
    # Pixel ratio for the pictures
    with open(filename, 'rb') as f:
        # Reads in from file here:
        for character in categories:
            # Skipping forward for each character:
            # ARGS:
            #   sheets      : The number of writers
            #   record_len  : The records length in bytes
            #--------------------------------------------

            f.seek((character * sheets + 1) * record_len)

            #feature_part, labels_part = save_data(f,record_len, sheets, size, database)
            # feature_part, labels_part = parse_data(f,record_len, sheets, size, database)


            # feature_part, labels_part = save_data(  f = f,
            #                                         record_len = record_len,
            #                                         num_char = 1,
            #                                         size = size,
            #                                         database = database,
            #                                         vectorize = vectorize,
            #                                         image_brightness = image_brightness)
            feature_part, labels_part = parse_data(f = f,
                                                    record_len = record_len,
                                                    num_char = sheets,
                                                    size = size,
                                                    database = database,
                                                    vectorize = vectorize,
                                                    image_brightness = image_brightness)
            feature += feature_part
            labels += labels_part
            #--------------------------------------------

    return feature, labels
