export TF_CPP_MIN_LOG_LEVEL=2
python material.py -src en -tgt sw -c en -m mt --phase train
python material.py -src en -tgt tl -c en -m mt --phase train
python material.py -src en -tgt sw -c sw -m fastext --phase train
python material.py -src en -tgt tl -c tl -m fastext --phase train
