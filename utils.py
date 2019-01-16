#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
import matplotlib.axes
import numpy as np
import os
import pandas as pd
import pytesseract

from fontTools.ttLib import TTFont
from nltk.metrics import edit_distance
from PIL import ImageFont, ImageDraw, Image


def get_font_name(fontfile):
    """
    Read font name from a font file.
    """
    font = TTFont(fontfile)
    for record in font["name"].names:
        if record.nameID == 4:
            return record.toUnicode()
        
        
def get_font_names(directory):
    """
    Read font names from all TTF and OTF files in a directory. 
    Return a pd.DataFrame with font file path, name and unique index number.
    """
    fontdata = []
    i = 0
    files = os.listdir(directory)
    for name in files:
        path = os.path.join(directory, name)
        if not (os.path.isfile(path) and name[-3:].lower() in ("otf","ttf")):
            continue
        data = {"id": i}
        data["path"] = path
        data["name"] = get_font_name(path)
        fontdata.append(data)
        i += 1
    return pd.DataFrame(fontdata).set_index("id")


def render_text_bw(text, fontfile, ptsize=20, linespacing=0, as_nparray=True):
    """
    Render text using given fontfile and params.
    Return a BLACK AND WHITE image either as np.array or as PIL.Image.
    """
    font = ImageFont.truetype(fontfile, ptsize)
    # Create a dummy image, draw text and find out its size 
    pimg = Image.new("L", (1, 1))
    draw = ImageDraw.Draw(pimg)
    x,y = draw.textsize(text, font=font, spacing=linespacing)
    # Create a new image large enough for the text
    pimg = Image.new("L", (x+10,y+15))
    draw = ImageDraw.Draw(pimg)
    draw.text((5,7), text, fill=1, font=font, spacing=linespacing)
    if as_nparray:
        return np.array(pimg,dtype=np.uint8)
    return pimg


def render_text_gray(text, fontfile, ptsize=20, linespacing=0, as_nparray=True):
    """
    Render text using given fontfile and params.
    Return a GRAYSCALE image either as np.array or as PIL.Image.
    """
    font = ImageFont.truetype(fontfile, ptsize)
    pimg = Image.new("RGB", (1, 1))
    draw = ImageDraw.Draw(pimg)
    x,y = draw.textsize(text, font=font, spacing=linespacing)
    pimg = Image.new("RGB", (x+10,y+15))
    draw = ImageDraw.Draw(pimg)
    draw.text((5,7), text, fill=(255,255,255), font=font, spacing=linespacing)
    if as_nparray:
        return np.array(pimg,dtype=np.uint8)[:,:,0]
    return pimg


def preview_fonts(fontdata, ids=None, ptsize=36):
    """
    Generate and show previews of fonts listed in fontdata pd.DataFrame.
    Show all, unless a list of specific IDs is given.
    """
    previews = []
    for row in fontdata.sort_values(by="name").itertuples():
        if ids and not getattr(row,"Index") in ids:
            continue
        img = render_text_gray(getattr(row,"name"), getattr(row,"path"), ptsize)
        previews.append(255-img)

    fig = plt.figure(figsize=(12, len(previews)))
    for i in range(len(previews)):
        fig.add_subplot(len(previews), 1, i+1)
        plt.axis('off')
        plt.imshow(previews[i], cmap='gray')
    plt.show()


def breaklines(text, char_limit=60):
    """
    Break text into lines so none is loner than char_limit.
    """
    settext = []
    for par in text.split("\n"):
        chunk = []
        chunk_len = 0
        for word in par.split(" "):
            if len(word) + chunk_len > char_limit:
                settext.append(" ".join(chunk))
                chunk = []
                chunk_len = 0
            chunk.append(word)
            chunk_len += len(word)
        splitpar = " ".join(chunk)
        if splitpar:
            settext.append(splitpar)
    return "\n".join(settext)


def make_experiment_setup(fontdata, ptsizelist, outdir):
    """
    Sets up experiment metadata(base) and returns it in a pd.DataFrame.
    """
    imgdir = os.path.join(outdir,"img")
    txtdir = os.path.join(outdir,"txt")
    os.makedirs(imgdir, exist_ok=True)
    os.makedirs(txtdir, exist_ok=True)
    
    setup = pd.DataFrame(columns="fontid ptsize img txt".split())
    
    for font in fontdata.itertuples():
        fontid = getattr(font,"Index")
        for ptsize in ptsizelist:
            img = os.path.join(imgdir,"{:02}.{}.png".format(fontid,ptsize))
            txt = os.path.join(txtdir,"{:02}.{}.txt".format(fontid,ptsize))
            setup = setup.append({
                "fontid": fontid, "ptsize": ptsize, "img": img, "txt": txt
            }, ignore_index=True)
            
    return setup


def batch_render_text(text, setup, fontdata):
    """
    Batch renders the source text using the settings specified by 
    the setup and fontdata DataFrames.
    """
    for row in setup.itertuples():
        fontfile = fontdata.iloc[getattr(row,"fontid")]["path"]
        filename = getattr(row,"img")
        ptsize = getattr(row,"ptsize")
        img = render_text_bw(text, fontfile, ptsize, linespacing=ptsize//2, as_nparray=False)
        img.save(filename)
        print("Wrote", filename)
         
            
def batch_ocr_images(setup, language="eng"):
    """
    Batch OCRs the generated images.
    """
    for row in setup.itertuples():
        imgname = getattr(row,"img")
        txtname = getattr(row,"txt")
        img = np.array(Image.open(imgname),dtype=np.uint8)
        txt = pytesseract.image_to_string(img, lang=language)
        txt = txt.replace("\n\n","\n")
        with open(txtname,"w") as out:
            out.write(txt)
        print("Wrote", txtname)


def WER(s1, s2):
    """
    Computes word error rate using Levenshtein distance.
    """
    tok1, tok2 = s1.split(), s2.split()
    return edit_distance(tok1, tok2)/max(len(tok1), len(tok2))


def CER(s1, s2):
    """
    Computes character error rate using Levenshtein distance.
    """
    tok1, tok2 = list(s1), list(s2)
    return edit_distance(tok1, tok2)/max(len(tok1), len(tok2))


def batch_evaluate(setup, fontdata, orig_text):
    """
    Evaluates OCR output files against source text using WER and CER.
    """
    results = pd.DataFrame(columns="Font id,Font,Point size,WER,CER,txt".split(","))
    for row in setup.itertuples():
        fontid = getattr(row,"fontid")
        font = fontdata.iloc[fontid]["name"]
        txtfile = getattr(row,"txt")
        ptsize = getattr(row,"ptsize")
        with open(txtfile, "r") as fin:
            ocr_text = fin.read()
            wer = WER(orig_text, ocr_text)
            cer = CER(orig_text, ocr_text)
            results.loc[results.shape[0]] = {
                "Font id": fontid, "Font": font, "Point size": ptsize, 
                "WER": wer, "CER": cer, "txt": txtfile
            }
        print("Evaluated", txtfile)
    return results

