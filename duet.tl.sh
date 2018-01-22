#!/bin/bash
for  i in 0.0 0.2 0.4 0.6 0.8
do
	python 5-fold.py --phase train --split $i --model_file examples/tl/config/duet_ranking_en.config # _en,_tl,_sw
	python matchzoo/main.py --phase train --model_file examples/tl/config/duet_ranking_en.config # _en,_tl,_sw
	python matchzoo/main.py --phase predict --model_file examples/tl/config/duet_ranking_en.config # _en, _tl,_sw
	mv "predict.test.duet_ranking.txt" "predict.$i.txt"
done

cat predict.0* > predict.test.duet_ranking.txt
rm predict.0*
