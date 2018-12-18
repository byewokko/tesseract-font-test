#!/usr/bin/env python
# coding: utf-8

import numpy as np              # Vector math
import pytesseract              # OCR
import matplotlib.pyplot as plt # Plotting
from PIL import ImageFont, ImageDraw, Image
from borrowed_code import wer

def breaklines(text, char_limit=60):
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
    
def render_text(texttorender, ptsize, fontfile):
    font = ImageFont.truetype(fontfile, ptsize)
    # Create a dummy image, draw text and find out its size 
    pimg = Image.new('L', (1, 1))
    draw = ImageDraw.Draw(pimg)
    x,y = draw.textsize(texttorender, font=font)
    # Create a new image large enough for the text
    pimg = Image.new('L', (x+10,y+14))
    draw = ImageDraw.Draw(pimg)
    draw.text((5,7), texttorender, fill=1, font=font)
    img = np.array(pimg,dtype=np.uint8)
    return img

def render_text_gray(texttorender, ptsize, fontfile):
    # To introduce the gray edges, we generate the text 
    # in a larger scale and then shrink it down.
    # We also need to use RGB, since Pillow does not have grayscale mode 
    rf = 4 #resize factor
    font = ImageFont.truetype(fontfile, rf*ptsize)
    pimg = Image.new("RGB", (1, 1))
    draw = ImageDraw.Draw(pimg)
    x,y = draw.textsize(texttorender, font=font)
    pimg = Image.new("RGB", (x+10*rf,y+15*rf))
    draw = ImageDraw.Draw(pimg)
    draw.text((5*rf,5*rf), texttorender, fill=(255,255,255), font=font)
    pimg_resized = pimg.resize((x//rf,y//rf), Image.BICUBIC)
    img = np.array(pimg_resized,dtype=np.uint8)[:,:,0]
    return img

file = "Eisenhower.txt"
#file = "zprava.txt"

# Read in some text
with open(file, 'r') as file:
    original_text = file.read()

original_text = breaklines(original_text)
print(original_text)


# ## Render text with a custom font

# In[106]:




#fontfile="/usr/share/fonts/dejavu/DejaVuSans-ExtraLight.ttf"
#fontfile="/usr/share/fonts/liberation/LiberationMono-BoldItalic.ttf"
#fontfile="/usr/share/fonts/dejavu/DejaVuSansMono-Bold.ttf"
#fontfile="GentiumPlus-R.ttf"
#fontfile="GentiumPlus-I.ttf"
#fontfile="KGFeeling22.ttf"
#fontfile="comicz.ttf"
fontfile="ocr-aregular.ttf" # no CZ characters
#fontfile="chfraktur.ttf" # no CZ characters

image_with_text = render_text(original_text, 18, fontfile=fontfile)
#image_with_text = render_text_gray(original_text, 18, fontfile=fontfile)

plt.figure(figsize=(12, 12))
plt.imshow(image_with_text, cmap='gray');


# Run Tesseract.

# In[108]:


extracted_text = pytesseract.image_to_string(image_with_text, lang="eng")
extracted_text = extracted_text.replace("\n\n","\n")
print(extracted_text)


# # Evaluation
# 
# A common metric of quality of the OCR is the word error rate, i.e. the number of non-recognised words in relation to the total number of words. This can be done by flexibly matching the original text with the text returned from the OCR.
# 
# A big thanks to https://martin-thoma.com/word-error-rate-calculation/ for sharing their code on Levenshtein distance.

# In[84]:


from borrowed_code import wer
#print(wer.__doc__)


# In[85]:


print(original_text.split()[:20])
print(extracted_text.split()[:20])
we = wer(original_text.split(), extracted_text.split())
print("Word errors:", we)
print("WER:", we/max(len(original_text.split()), len(extracted_text.split())))


# In[86]:


print(list(original_text)[:20])
print(list(extracted_text)[:20])
ce = wer(list(original_text), list(extracted_text))
print("Character errors:", ce)
print("CER:", ce/max(len(list(original_text)), len(list(extracted_text))))

