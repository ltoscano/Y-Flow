cd ../../

currpath=`pwd`
# train the model
python yflow/main.py --phase train --model_file ${currpath}/examples/wikiqa/config/duet_wikiqa.config


# predict with the model

python yflow/main.py --phase predict --model_file ${currpath}/examples/wikiqa/config/duet_wikiqa.config
