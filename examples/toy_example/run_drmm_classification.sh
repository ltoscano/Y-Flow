cd ../../

currpath=`pwd`
# train the model
python yflow/main.py --phase train --model_file ${currpath}/examples/toy_example/config/drmm_classify.config


# predict with the model

python yflow/main.py --phase predict --model_file ${currpath}/examples/toy_example/config/drmm_classify.config
