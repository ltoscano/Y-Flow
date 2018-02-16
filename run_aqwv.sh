python calculate_aqwv.py --predict results/src_en_-tgt_sw_-c_sw_-m_wiktionary_--phase_train.txt_predict --judge data/sw/judg/rel.judg --total 471 --language sw > sw_aqwv.txt
python calculate_aqwv.py --predict results/src_en_-tgt_sw_-c_sw_-m_google_--phase_train.txt_predict --judge data/sw/judg/rel.judg --total 471 --language sw > sw_aqwv.txt
python calculate_aqwv.py --predict results/src_en_-tgt_sw_-c_en_-m_mt_--phase_predict.txt_predict --judge data/sw/judg/rel.judg --total 471 --language sw > sw_aqwv.txt
python calculate_aqwv.py --predict results/src_en_-tgt_sw_-c_sw_-m_fastext_--phase_predict.txt_predict --judge data/sw/judg/rel.judg --total 471 --language sw > sw_aqwv.txt

#python calculate_aqwv.py --predict results/src_en_-tgt_tl_-c_en_-m_mt_--phase_predict.txt_predict --judge data/tl/judg/rel.judg --total 462 --language tl > tl_aqwv.txt
