<div align='center'>
<img src="./docs/_static/images/tree-shot.png" width = "600"  alt="tree-shot" align=center />
</div>

Y-Flow is an extension for MatchZoo <https://github.com/faneshion/MatchZoo>  toolkit for text matching. Here at Yale we are developing a system for text matching for question answering, document ranking, paraphrase identification, and machine translation. The figure above indicates a surprise language coming to our system and we are going to use zero-shot learning approaches for transfer learning particularly for syntax. 

<table>
  <tr>
    <th width=30%, bgcolor=#999999 >Tasks</th> 
    <th width=20%, bgcolor=#999999>Text 1</th>
    <th width="20%", bgcolor=#999999>Text 2</th>
    <th width="20%", bgcolor=#999999>Objective</th>
    <th width="20%", bgcolor=#999999>author</th>
  </tr>
    <tr>
    <td align="center", bgcolor=#eeeeee> Transfer Learning </td>
    <td align="center", bgcolor=#eeeeee> ranking 1 </td>
    <td align="center", bgcolor=#eeeeee> ranking 2 </td>
    <td align="center", bgcolor=#eeeeee> ranking </td>
    <td align="center", bgcolor=#eeeeee> yflow </td>
  </tr>
  <tr>
    <td align="center", bgcolor=#eeeeee> Paraphrase Indentification </td>
    <td align="center", bgcolor=#eeeeee> string 1 </td>
    <td align="center", bgcolor=#eeeeee> string 2 </td>
    <td align="center", bgcolor=#eeeeee> classification </td>
    <td align="center", bgcolor=#eeeeee> matchzoo </td>
  </tr>
  <tr>
    <td align="center", bgcolor=#eeeeee> Sentiment Analysis </td>
    <td align="center", bgcolor=#eeeeee> product </td>
    <td align="center", bgcolor=#eeeeee> review </td>
    <td align="center", bgcolor=#eeeeee> classification </td>
    <td align="center", bgcolor=#eeeeee> yflow </td>
  </tr>
  <tr>
    <td align="center", bgcolor=#eeeeee> Sentence Matching </td>
    <td align="center", bgcolor=#eeeeee> sentence 1 </td>
    <td align="center", bgcolor=#eeeeee> sentence 2 </td>
    <td align="center", bgcolor=#eeeeee> score </td>
     <td align="center", bgcolor=#eeeeee> yflow </td>

  </tr>
  <tr>
    <td align="center", bgcolor=#eeeeee> Textual Entailment </td>
    <td align="center", bgcolor=#eeeeee> text </td>
    <td align="center", bgcolor=#eeeeee> hypothesis </td>
    <td align="center", bgcolor=#eeeeee> classification </td>
     <td align="center", bgcolor=#eeeeee> matchzoo </td>

  </tr>
  <tr>
    <td align="center", bgcolor=#eeeeee> Question Answer </td>
    <td align="center", bgcolor=#eeeeee> question </td>
    <td align="center", bgcolor=#eeeeee> answer </td>
    <td align="center", bgcolor=#eeeeee> classification/ranking </td>
     <td align="center", bgcolor=#eeeeee> matchzoo </td>

  </tr>
  <tr>
    <td align="center", bgcolor=#eeeeee> Conversation </td>
    <td align="center", bgcolor=#eeeeee> dialog </td>
    <td align="center", bgcolor=#eeeeee> response </td>
    <td align="center", bgcolor=#eeeeee> classification/ranking </td>
     <td align="center", bgcolor=#eeeeee> matchzoo </td>
  </tr>
  <tr>
    <td align="center", bgcolor=#eeeeee> Information Retrieval </td>
    <td align="center", bgcolor=#eeeeee> query </td>
    <td align="center", bgcolor=#eeeeee> document </td>
    <td align="center", bgcolor=#eeeeee> ranking </td>
      <td align="center", bgcolor=#eeeeee> yflow </td>
  </tr>
</table>

## Installation
Please clone the repository and run
```
conda install -c anaconda tensorflow-gpu
git clone https://github.com/javiddadashkarimi/Y-Flow.git
cd Y-Flow
python setup.py install
export TF_CPP_MIN_LOG_LEVEL=2
```
In the main directory, this will install the dependencies automatically.
Or run the following to run the dependencies:`pip install -r requirements.txt`.

Then install [trec_eval](http://trec.nist.gov/trec_eval/) for runnning the system.
Visit [IndriBuildIndex](https://www.lemurproject.org/lemur/indexing.php#IndriBuildIndex) to install IndriBuildIndex application.

### Data Preparation
Different text matching formats are considered in this project for unification:

+	**Word Dictionary**: records the mapping from each word to a unique identifier called *wid*. Words that are too frequent (e.g. stopwords), too rare or noisy (e.g. fax numbers) can be  filtered out by predefined rules.
+	**Corpus File**: records the mapping from each text to a unique identifier called *tid*, along with a sequence of word identifiers contained in that text. Note here each text is truncated or padded to a fixed length customized by users.
+	**Relation File**: is used to store the relationship between two texts, each line containing a pair of *tids* and the corresponding label.
+   **Detailed Input Data Format**: a detailed explaination of input data format can be found in Y-Flow/data/example/readme.md.
+   **Preprocessing for the CLEF data**: The [CLEF](http://catalog.elra.info/en-us/repository/browse/clef-adhoc-news-test-suites-2004-2008-evaluation-package/4bc886f4a9e711e7a093ac9e1701ca02f0ac3dffb8164c3e8ff5f0dcfa7d33dd/) comes with multiple xml files, containing multiple documents. We provide a preprocessor in ``tools/``.
+  **Example**: for indexing and also generating training data in document ranking: 
```
  - IndriBuildIndex index.param
  - sh generate_ranking_data.sh
  - python3 tools/xml2text.py data/fr/CLEF-FR/More_French/lemonde95/ data/fr/CLEF-FR/docs
```
Notice that we use python3 to avoid encoding errors.

### Model Construction
In the model construction module, we employ Keras library to help users build the deep matching model layer by layer conveniently. The Keras libarary provides a set of common layers widely used in neural models, such as convolutional layer, pooling layer, dense layer and so on. To further facilitate the construction of deep text matching models, we extend the Keras library to provide some layer interfaces specifically designed for text matching.

Moreover, the toolkit has implemented two schools of representative deep text matching models, namely representation-focused models and interaction-focused models [[Guo et al.]](http://www.bigdatalab.ac.cn/~gjf/papers/2016/CIKM2016a_guo.pdf).

### Training and Evaluation
For learning the deep matching models, the toolkit provides a variety of objective functions for regression, classification and ranking. For example, the ranking-related objective functions include several well-known pointwise, pairwise and listwise losses. It is flexible for users to pick up different objective functions in the training phase for optimization. Once a model has been trained, the toolkit could be used to produce a matching score, predict a matching label, or rank target texts (e.g., a document) against an input text.

## Usage:

For the document ranking example, you can run
```
# to trigger machine translation-involved version:
python material.py -src en -tgt sw -c en -m mt

# to trigger google translation-involoved version:
python material.py -src en -tgt tl -c tl -m google

# add query list by -q to show detailed results:
python material.py -src en -tgt tl -c tl -m google -q query991 query306

```
    -'--source','-src', default='en', help='source language [sw,tl,en]'
    -'--target','-tgt', default='sw', help='target language [sw,tl,en]'
    -'--collection','-c', default='en', help='language of documents [sw,tl,en]'
    -'--out','-o', default='en', help='output language [sw,tl,en]'
    -'--method','-m', default='mt', help='method [mt,google,wiktionary,fastext]'
    -'--query_list', '-q', nargs='*',help='check query result [query Id list]'



## Model Detail:

1. DRMM : this model is an implementation of <a href="http://www.bigdatalab.ac.cn/~gjf/papers/2016/CIKM2016a_guo.pdf">A Deep Relevance Matching Model for Ad-hoc Retrieval</a>.

2. MatchPyramid : this model is an implementation of <a href="https://arxiv.org/abs/1602.06359"> Text Matching as Image Recognition</a>

3. ARC-I : this model is an implementation of <a href="https://arxiv.org/abs/1503.03244">Convolutional Neural Network Architectures for Matching Natural Language Sentences</a>

4. DSSM : this model is an implementation of <a href="https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/cikm2013_DSSM_fullversion.pdf">Learning Deep Structured Semantic Models for Web Search using Clickthrough Data</a>

5. CDSSM : this model is an implementation of <a href="https://www.microsoft.com/en-us/research/publication/learning-semantic-representations-using-convolutional-neural-networks-for-web-search/">Learning Semantic Representations Using Convolutional Neural Networks for Web Search</a>

6. ARC-II : this model is an implementation of <a href="https://arxiv.org/abs/1503.03244">Convolutional Neural Network Architectures for Matching Natural Language Sentences</a>

7. MV-LSTM : this model is an implementation of <a href="https://arxiv.org/abs/1511.08277">A Deep Architecture for Semantic Matching with Multiple Positional Sentence Representations</a>

8. aNMM : this model is an implementation of <a href="http://maroo.cs.umass.edu/pub/web/getpdf.php?id=1240">aNMM: Ranking Short Answer Texts with Attention-Based Neural Matching Model</a>


9. DUET : this model is an implementation of <a href="https://dl.acm.org/citation.cfm?id=3052579">Learning to Match Using Local and Distributed Representations of Text for Web Search</a>

10. models under development:

<a href="https://arxiv.org/abs/1604.04378">Match-SRNN</a>, <a href="https://arxiv.org/abs/1710.05649">DeepRank</a>, <a href="https://arxiv.org/abs/1706.06613">K-NRM</a> ....

Development Teams
====
| author | affiliation | task | page |
| --- | --- | --- | --- |
|**Irene Li**|  Yale University | Information Retrieval | [GitHub](https://github.com/IreneZihuiLi)|
|**Caitlin Westerfield**|  Yale University | Transfer Learning | [GitHub](https://github.com/CMWesterfield16)|
|**Gaurav Pathak**|  Yale University | Zero-shot Learning | [GitHub](https://github.com/GauravPathakYale)|
|**Javid Dadashkarimi**|  Yale University | Organizer | [GitHub](https://github.com/dadashkarimi)|
|**Dragomir Radev**|  Yale University | Adviser | [GitHub](https://github.com/Yale-LILY)|

## Environment
* python2.7+
* tensorflow 1.4.1+
* keras 2.1.3+
* nltk 3.2.2+
* tqdm 4.19.4+
* h5py 2.7.1+
* indri 5.7
