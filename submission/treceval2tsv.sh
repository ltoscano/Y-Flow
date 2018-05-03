#python treceval2tsv.py --result_file results/src_en_-tgt_sw_-c_sw_-m_wiktionary.txt_predict --topic_file data/sw/topics/en.t 

#python treceval2tsv.py --result_file data/sw/EVAL-SW/result/result_cross_k7 --topic_file  /data/corpora/azure2/data/NIST-data/1A/IARPA_MATERIAL_BASE-1A/QUERY1/query_list.tsv --domain_file /data/corpora/azure2/data/NIST-data/1A/IARPA_MATERIAL_BASE-1A/QUERY1/domain_map.tsv --tsv_dir ./EVAL-SW-tsv
#python treceval2tsv.py --result_file data/sw/EVAL-SW/result/result.full --topic_file  /data/corpora/azure2/data/NIST-data/1A/IARPA_MATERIAL_BASE-1A/QUERY1/query_list.tsv --domain_file /data/corpora/azure2/data/NIST-data/1A/IARPA_MATERIAL_BASE-1A/QUERY1/domain_map.tsv --tsv_dir ./EVAL-SW-tsv-full
#
#python treceval2tsv.py --result_file data/tl/EVAL-TL/result/result_cross_k9 --topic_file  /data/corpora/azure2/data/NIST-data/1B/IARPA_MATERIAL_BASE-1B/QUERY1/query_list.tsv --domain_file /data/corpora/azure2/data/NIST-data/1B/IARPA_MATERIAL_BASE-1B/QUERY1/domain_map.tsv --tsv_dir ./EVAL-TL-tsv
#python treceval2tsv.py --result_file data/tl/EVAL-TL/result/result.full --topic_file  /data/corpora/azure2/data/NIST-data/1B/IARPA_MATERIAL_BASE-1B/QUERY1/query_list.tsv --domain_file /data/corpora/azure2/data/NIST-data/1B/IARPA_MATERIAL_BASE-1B/QUERY1/domain_map.tsv --tsv_dir ./EVAL-TL-tsv-full

#python treceval2tsv.py --topic_file  /data/corpora/azure2/data/NIST-data/1A/IARPA_MATERIAL_BASE-1A/QUERY1/query_list.tsv --annotation_file /data/corpora/azure2/data/NIST-data/1A/IARPA_MATERIAL_BASE-1A/ANALYSIS_ANNOTATION1/query_annotation.tsv --domain_file /data/corpora/azure2/data/NIST-data/1A/IARPA_MATERIAL_BASE-1A/QUERY1/domain_map.tsv --tsv_dir ./EVAL-SW-tsv
#python treceval2tsv.py --topic_file  /data/corpora/azure2/data/NIST-data/1A/IARPA_MATERIAL_BASE-1A/QUERY1/query_list.tsv --annotation_file /data/corpora/azure2/data/NIST-data/1A/IARPA_MATERIAL_BASE-1A/DEV_ANNOTATION1/query_annotation.tsv --domain_file /data/corpora/azure2/data/NIST-data/1A/IARPA_MATERIAL_BASE-1A/QUERY1/domain_map.tsv --tsv_dir ./EVAL-SW-tsv

#python treceval2tsv.py --topic_file  /data/corpora/azure2/data/NIST-data/1B/IARPA_MATERIAL_BASE-1B/QUERY1/query_list.tsv --annotation_file /data/corpora/azure2/data/NIST-data/1B/IARPA_MATERIAL_BASE-1B/ANALYSIS_ANNOTATION1/query_annotation.tsv --domain_file /data/corpora/azure2/data/NIST-data/1B/IARPA_MATERIAL_BASE-1B/QUERY1/domain_map.tsv --tsv_dir ./EVAL-TL-tsv-full
#python treceval2tsv.py --topic_file  /data/corpora/azure2/data/NIST-data/1B/IARPA_MATERIAL_BASE-1B/QUERY1/query_list.tsv --annotation_file /data/corpora/azure2/data/NIST-data/1B/IARPA_MATERIAL_BASE-1B/DEV_ANNOTATION1/query_annotation.tsv --domain_file /data/corpora/azure2/data/NIST-data/1B/IARPA_MATERIAL_BASE-1B/QUERY1/domain_map.tsv --tsv_dir ./EVAL-TL-tsv-full

#python treceval2tsv.py --topic_file  /data/corpora/azure2/data/NIST-data/1B/IARPA_MATERIAL_BASE-1B/QUERY2/query_list.tsv --domain_file /data/corpora/azure2/data/NIST-data/1B/IARPA_MATERIAL_BASE-1B/QUERY2/domain_map.tsv --tsv_dir ./EVAL12_tl_dt_phrase_cutoff100 --result_file 20180421/result.file.tl.dt.phrase.cutoff100
#python treceval2tsv.py --topic_file  /data/corpora/azure2/data/NIST-data/1B/IARPA_MATERIAL_BASE-1B/QUERY2/query_list.tsv --domain_file /data/corpora/azure2/data/NIST-data/1B/IARPA_MATERIAL_BASE-1B/QUERY2/domain_map.tsv --tsv_dir ./EVAL12_tl_qt_phrase_cutoff100 --result_file 20180421/result.file.tl.qt.phrase.cutoff100
#
#python treceval2tsv.py --topic_file  /data/corpora/azure2/data/NIST-data/1A/IARPA_MATERIAL_BASE-1A/QUERY2/query_list.tsv --domain_file /data/corpora/azure2/data/NIST-data/1A/IARPA_MATERIAL_BASE-1A/QUERY2/domain_map.tsv --tsv_dir ./EVAL12_sw_qt_phrase_cutoff100 --result_file 20180421/result.file.sw.qt.phrase.cutoff100
#python treceval2tsv.py --topic_file  /data/corpora/azure2/data/NIST-data/1A/IARPA_MATERIAL_BASE-1A/QUERY2/query_list.tsv --domain_file /data/corpora/azure2/data/NIST-data/1A/IARPA_MATERIAL_BASE-1A/QUERY2/domain_map.tsv --tsv_dir ./EVAL12_sw_dt_phrase_cutoff100 --result_file 20180421/result.file.sw.dt.phrase.cutoff100

python treceval2tsv.py --topic_file  /data/corpora/azure2/data/NIST-data/1B/IARPA_MATERIAL_BASE-1B/QUERY1/query_list.tsv --domain_file /data/corpora/azure2/data/NIST-data/1B/IARPA_MATERIAL_BASE-1B/QUERY1/domain_map.tsv --tsv_dir ./ANALYSIS_tl --result_file 20180421/result.file

#python treceval2tsv.py --topic_file  /data/corpora/azure2/data/NIST-data/1A/IARPA_MATERIAL_BASE-1A/QUERY2/query_list.tsv --domain_file /data/corpora/azure2/data/NIST-data/1A/IARPA_MATERIAL_BASE-1A/QUERY2/domain_map.tsv --tsv_dir ./EVAL12_sw --result_file 20180421/result.file