This program utilizes Google Sheet's translation function to translate word from a source language to a target language.

Currently, Google Sheet's translation function, i.e., =GOOGLETRANSLATE(text, [source_language], [target_language]), supports 105 different languages:

af, am, ar, az, be, bg, bn, bs, ca, co, cs, cy, da, de, el, en, eo, es, et, eu, fa, fi, fr, fy, ga, gd, gl, gu, ha, he, hi, hr, ht, hu, hy, id, ig, in, is, it, iw, ja, ji, jv, jw, ka, kk, km, kn, ko, ku, ky, la, lb, lo, lt, lv, mg, mi, mk, ml, mn, mr, ms, mt, my, nb, ne, nl, no, ny, pa, pl, ps, pt, ro, ru, sd, si, sk, sl, sm, sn, so, sq, sr, st, su, sv, sw, ta, te, tg, th, tl, tr, uk, ur, uz, vi, xh, yi, yo, zh, zu


## Installation ##

Please run the command

```
pip install --upgrade google-api-python-client
```

## Usage ##

Please run the command to translate all the words listed in a txt file

```
python translate.py [source_language] [target_language] [txt file]
```

For example

```
python translate.py en fr words.txt
```

The translation will be printed to the command line in the following format

```
$ python translate.py en fr words.txt
natural language processing | traitement du langage naturel
translation | Traduction
apple | Pomme
chocolate | Chocolat
```

### txt file format ###

The words should be listed one word per line in the txt file.


