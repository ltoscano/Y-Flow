cd ../../

currpath=`pwd`
# train the model
python yflow/main.py --phase train --model_file ${currpath}/examples/wikiqa/config/cdssm_wikiqa.config


# predict with the model

python yflow/main.py --phase predict --model_file ${currpath}/examples/wikiqa/config/cdssm_wikiqa.config
