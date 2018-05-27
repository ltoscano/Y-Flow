cd ../../

currpath=`pwd`
# train the model
python yflow/main.py --phase train --model_file ${currpath}/examples/toy_example/config/matchpyramid_ranking.config


# predict with the model

python yflow/main.py --phase predict --model_file ${currpath}/examples/toy_example/config/matchpyramid_ranking.config
