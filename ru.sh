export TF_CPP_MIN_LOG_LEVEL=2
python material.py -src en -tgt sw -c sw -m google #en -> sw query translation using Google
python material.py -src en -tgt sw -c sw -m wiktionary #en -> sw query translation using wiktionary
#python material.py -src en -tgt sw -c en -m mt     #sw -> document translation using machine translation; still using DUET

python material.py -src en -tgt tl -c tl -m google #en -> tl query translation using Google
python material.py -src en -tgt tl -c tl -m wiktionary #en -> tl query translation using wiktionary
