# A Multi-View Deep Learning Approach for Cross Domain User Modeling in Recommendation System

## Ali Elkahky, Yang Song, Xiaodong He,  WWW'15 [PDF](http://research.microsoft.com/pubs/238334/frp1159-songA.pdf)

## Abstract

近年のオンラインサービスは大量のユーザに対して、関連のあるコンテンツを推薦するために、personalizeする事が大事である。その時に必要となるのは、コールドスタート問題（新規ユーザ）に対応することである。本研究では、推薦の質とスケーラビリティの両方を実現する内容ベース推薦システムを提案する。

本研究では、ユーザのウェブ上でのブラウジング履歴や検索クエリを用いて、ユーザをリッチに表現する。また、ユーザとそのユーザが好むアイテム間の関連性を最大化するような潜在空間にユーザとアイテムをマップするためにDeep learningを用いた。Multi-view deep learning modelを導入することで、異なるドメインのアイテム特徴とユーザ特徴から一緒に学習ができる。

我々は、入力特徴の次元と訓練データの量を削減することによってユーザ表現に基づくリッチな特徴をスケーラブルに作成する方法を提案した。リッチなユーザ特徴表現は、関連のあるユーザの行動パターンを学習し，有用な推薦を（まだ何もサービスとインタラクションがない新規の）ユーザに与えることを可能にする。異なるドメインを組み合わせて単一のモデルで学習することは、一つのドメインを超えたあらゆるドメインにおいて、よりコンパクトで豊かなセマンティックユーザ潜在特徴ベクトルを作成可能にし、推薦の質の向上を導く。

実際に以下の3つのレコメンデーションシステムで評価実験した。

 1. Windows Apps recommendation
 2. News recommendation
 3. Movie/TV recommendation

 
実験により，提案手法がstate-of-the-artアルゴリズムより優れていることを確認した（既存ユーザに対して49%向上, 新規ユーザに対して115%向上）。また、パブリックに公開されているデータセットでも実験を行い、クロスドメインのレコメンデーションシステムを構成する上で、トラディショナルなトピックモデルの手法より優れていることを確認した。

Scalabilityの点に関しては、提案手法は数100万規模のユーザと数10億規模のアイテムに対して容易にスケールできる。全てのドメインから得られる特徴を組み合わせることは、それぞれのドメインにモデルを分けてやるより良いパフォーマンスであることを実験で確認した。

## Gneral Terms
User Modeling

## Key words
User Modeling, Recommendation System, Multi-view learning, Deep learning

## 1. Introduction

web serviceにおいて、推薦システムとpersonalizationのの重要性は増してきている。プロバイダは、ユーザと相関の高いアイテムを出来る限り早く見つけたい。



### 推薦システムの主要なアプローチ

```
- 協調フィルタリング(Collaborative filtering: CF)
	- ユーザのアイテム閲覧履歴(や評価)の類似性をもとに推薦
- 内容ベースフィルタリング(Content-based filtering: CB)
	- ユーザやアイテムの特徴の類似性をもとに推薦
```


両方アプリケーションとして一定の成果上げてる。Challengesはpersonalizationと推薦の質を両立することである。CFは履歴をもとに推薦するから、質の高い推薦するためにはユーザのアイテムへの閲覧履歴がたくさん必要になる。この問題は**cold start problem**として知られている。新規のweb serviceでは履歴がないのでこのcold start problemがより顕著である。よって、トラディショナルなCFは新規ユーザに対してはいい推薦できない。

逆に、CBは推薦するためにユーザやアイテムの特徴を使う。例えば、新しい記事$N_i, N_j$が同じトピックを共有しており、ユーザが$N_i$を好きな場合、システムは$N_j$をユーザに推薦できる。同様に、ユーザ$U_i, U_j$が住んでいる場所や年齢、性別などが類似している場合、システムは$U_i$が過去にlikeしたものを$U_j$に推薦できる（これは履歴を使っているけど、CFと異なるのは、ユーザの特徴の類似性から推薦しているとこ。CFはユーザの履歴の類似性から推薦する）。

実際に研究でも、CBはcold start problemに対応可能だと示されている。**しかし実際はその有効性には疑問がある。なぜなら、ユーザの特徴を限られた情報から獲得することは一般に難しく、だいたいは実際のユーザの興味を正確に捉えることはできない。**このような限界に挑戦するために、我々は、ユーザとアイテムの両方をうまく利用した推薦システムを提案する。

そこで、本研究では、ユーザのprofileを用いるのではなく、ユーザのブラウジングやサーチの履歴からユーザの興味をモデル化して、リッチなユーザ表現を獲得する手法を提案する。根本的な仮定として、ユーザのオンライン上での行動はユーザのバックグラウンドや嗜好を反映するので、ユーザが興味があるであろうアイテムやトピックの正確な洞察を得ることができる。

例えば、幼児に関連する検索クエリやサイト(トイザらスなど)の訪問は、彼女が幼児の母であることを提案できるだろう。潤沢にあるユーザのオンライン上の行動を用いることで、より効率的かつ効果的な推薦ができる。

我々の研究では、Deep Structured Semantic Models (DSSM)を拡張して、ユーザとアイテムを共通のsemantic spaceにマップし、そのマップした潜在空間上で最もそのユーザと相関が高いアイテムを推薦する。我々のモデルでは、ユーザとアイテムを、非線形変換を繰り返すことで、コンパクトなshare latent semantic space（射影後のユーザと、射影後の(ユーザが興味を示した)アイテムの相関が最大になるような空間）に射影する。

例えば、fifa.comに訪れた人はワールドカップのニュースを読むことや、Xboxでサッカーゲームをすることを好むであろうというようなマッピングを学習する。ユーザ側の豊かな特徴は、ユーザの行動をモデリングすることを可能にするので、従来のCBの多くの制限に打ち勝つことができる。もちろんユーザのcold start problemにも対応可能である。

たとえば、ユーザが音楽サービス上で何の履歴もなかったとしても、我々のモデルによってユーザの興味が得られていれば、音楽に関連するアイテムを推薦できる。我々のdeep learningモデルは、ranking-basedの目的をもっており、これは推薦システムに有効である。

さらに、single-view のDNNであるオリジナルのDSSMモデルを異なるドメインのアイテムから同時に学習するMulti-viewモデル（MV-DNN）に拡張した。共有の特徴空間を共有せずに学習するmulti-view learningはよく研究されている。我々はMV-DNNをmulti-view learningにおける一般的なdeep learningのアプローチとして考えた。特に、データセット（news, app, movie/tv logs）を用いて、各々のドメイン内でユーザ特徴からアイテム特徴にマップするseparateなモデルを構築する代わりに、全てのドメインのアイテムの特徴量を共通に最適化するmulti-view modelを構築した。

MV-DNNは全てのドメインのユーザ嗜好データを用いているので、ドメインを超えて有効なユーザ表現を学習できる上、data sparsity problemにも対応することができる。MV-DNNモデルは実験により、全てのドメインにおいて推薦の質を向上させたことを確認した。さらに、非線形マッピングにより、コンパクトなユーザ表現を得ることができ、これは、様々なタスクに使える。

Deep learningを使ってリッチなユーザ表現を得ることのChallengeは特徴空間が高次元になってしまい学習が難しくなることである。本研究では、scalabilityを実現するため、様々な次元削減技術を提案している。

本研究のコントリビューションは以下の5つである。

1. recommendation systemを構築するため、リッチなユーザ特徴を用いた
2. deep learningを用いたcontent-based recommendation systemを提案し、scalableなシステムを実現するための様々なテクニックを調査した
3. 複数のドメインを組み合わせるMulti-view deep learningモデルを導入し、recommendation systemを構築した
4. MV-DNNから学習されたsemantic feature mappingによりユーザのcold start problemに対応した
5. 実世界のビッグデータを用いた厳密な評価実験により、state-of-the-artを大きく上回る有効性を確認した



## 2. Related work

省略
 
## 3. Description of the data sets

以下の4つのデータセットを用いた。

1. Bingから得られる検索ログ
2. Bing Newsから得られるニュース記事のブラウジングログ
3. Windows AppStoreから得られるAppのダウンロードログ
4. Xboxから得られるMovie/TVの視聴ログ

すべてのログは、英語を公用語とするマーケット（US, Canada, UK）で2013.12-2014.6の間に収集された。

### user feature
- Bing上でのユーザの検索クエリとクリックしたURLを収集
- クエリはまず正規化され、unigram features（つまり一つ一つの独立した単語特徴）に分割した。URLは次元削減のためdomain-levelに短縮した（例えば、www.linkedin.comなど）
- 最も人気で価値のある特徴だけを保つために、TF-IDFのスコアを用いた
- 全部で、300万のunigram featuresと50万のdomain featuresを獲得し、ユーザ特徴として、合計350万の特徴ベクトルをえた

### news feature
- Bing Newsからニュース記事のクリックログを収集
- それぞれのアイテム（ニュース記事）は、以下の3つの特徴で表現されるnews featureを10万えた

1. title feature：文字のtri-gram表現にencode
2. top-level category：（ex, Entertainment）バイナリ特徴にencode
3. named entity（固有表現）：NLPパーサーで抽出。文字のtri-gram表現にencode

### app feature
- Windows AppStoreにおけるユーザのアプリダウンロード履歴を収集
- アプリのタイトルは文字のtri-gram表現にencodeし、バイナリ特徴にencodeしたカテゴリと結合
- アプリの説明文はよく変更されるので、特徴にいれなかった
- 5万の特徴ベクトルをえた

### movie/TV feature
- Xboxの映画やテレビの視聴履歴を収集
- タイトルと説明文はtext featureに結合され文字のtri-gram表現にencode
- ジャンルは例のごとくバイナリ特徴にencode
- 5万の特徴ベクトルをえた

我々のモデルでは、ユーザ特徴はuser viewにマップされ、それ以外の特徴はそれぞれ異なるitem viewにマップされる。モデルを学習するために、それぞれのuser viewはそのユーザを含むitem viewにマッチさせた。これを実現するためIDによって内部で結合し、user-item viewのペアからログインしているユーザをサブサンプリングした。これによってつくたれたデータセットはTable 1

![table1](/Users/Corpy/Desktop/table1.png)

## 4. DSSM for user modeling in recommendation systems

CIKM'13の論文[1]でweb検索において文書とクエリのマッチングを強化するDSSMモデルを提案した。今回紹介するmulti-viewモデルはそれに近い。DSSM（figure 1）を以下に簡単に説明する。

![fig1](/Users/Corpy/Desktop/fig1.png)

DNNへの入力(生のtext features)は高次元のterm vector（ex.正規化をしていないクエリやドキュメント内の出現回数）である。
DSSMでは入力はそれぞれ２つのNNを通過し、共有潜在空間内のsemantic vectorにマップされる。
web document rankingにおいては、DSSMは、クエリとドキュメントの相関をそれぞれ対応するsemantic vectorのコサイン類似度で求め、そのスコアによって、クエリをランキングする。

より正確に言うと、$x$がinput term vector、$y$がoutput vector、$l_i, i = 1, ..., N − 1$を中間の隠れ層、$W_i$を$i$番目の重み行列、$b_i$を$i$番目のbias termとすると、 

$$l_1 = W_1x$$
$$l_i = f(W_il_{i−1} +b_i), i=2,...,N−1$$ 
$$y = f(W_Nl_{N−1} + b_N)$$

となる。output layerとhidden layer$l_i, i = 2, ..., N − 1$の活性化関数としてシグモイド関数$tanh$を用いる。

$$f(x) = \frac{1 − e^{−2x}}{1+e^{-2x}}$$

クエリ$Q$とドキュメント$D$のsemantic relevance scoreはコサイン類似度で、

$$R(Q,D) = cosine(y_Q,y_D) = \frac{{y_Q}^T y_D}{ ||y_Q || · ||y_D ||}$$

のように求められる。
$y_Q$と$y_D$はそれぞれクエリとドキュメントのsemantic vectors。web searchでは、クエリが与えられた時、ドキュメントはこのrelevance scoreが高い順にソートして表示される。

従来は、それぞれ単語$w$はone-hot vectorで表現され、ベクトルの次元数はvocabularyのサイズだった。しかし実際の検索タスクでは、volcabularyのサイズは往々にしてとても大きいので、このone-hot vectorでモデルを学習するのはとてもハイコストだった。したがって、DSSMでは単語をletter-tri-gram vectorによって表現するためにword hashing layerを使う。

例えば、webという単語が与えられ、#web#のように単語の境界記号が与えられたとすると、単語はletter-n-gramに分割できる（ここではletter-tri-gramなので#-w-e, w-e-b, e-b-#）そして、単語はletter-tri-gramのcount vectorとして表現される。


例えば、Figure 1において、, 一層目の行列$W_1$はterm vectorからletter-tri-gram count vectorに変換する行列であり、学習は必要ない。全単語数が極端に大きくなったとしても、全letter-tri-gram数は限られている。それゆえ、これは訓練データにない新規単語に対しても一般化できる。

訓練では、クエリはそのクエリによってクリックされた文書
In training, it is assumed that a query is relevant to the documents that are clicked on for that query, and the parameters of the DSSM, 

すなわち、重み行列$W_i$は、これを用いて学習される。すなわち、あるクエリが与えられたときのドキュメントの事後確率は以下のsoftmax関数によって推定される。

$$P(D|Q) = \frac{exp(γR(Q, D))}{\sum_{D′∈ \bf{D}}{} exp(γR(Q,D′))}$$

γはsoftmax関数のsmoothing factorで、一般に実験的に求められる。$\bf{D}$は、ランキングされる文書の候補setを指す。

クエリとクリックされた文書のペアは$(Q, D^+)$、$Q$はクエリ、$D^+$はクリックされた文書。$\bf{D}$を$D^+$と$N$個のランダムに選ばれたクリックされていない文書$\{D_j^− ; j = 1, , N \}$で近似する。訓練では、model parameterはtrain set内でクエリが与えられたときクリックされた文書の確率を最大化するように推定される。つまり以下を最小化する。

$$L(Λ) = - log\prod_{(Q, D^+)}P(D^+|Q)$$

$Λ$はparameter set。

## 5. Multi-view deep neural network
MV-DNNはDSSMを2つ以上の異なるviewを一つの共有viewにマップするmulti-vieモデルに拡張したものであり、２つ以上の異なるviewの共有mapping viewを学習する際の一般的なモデルである（figure 2）。

![fig2](/Users/Corpy/Desktop/fig2.png)

このセッティングにおいて、viewは$v+1$個あり、pivot viewを$X_u$、それ以外を$X_1,...,X_v$とする。それぞれviewは非線形マッピング層$f_i(X_i, W_i)$を持ち、これらは、shared semantic space上の$Y_i$に変換される。

訓練データの、$j$番目のサンプルはpivot viewの$X_{u,j}$と、補助的なactive viewの$X_{a,j}$をもつ。ここでaはacitive viewのindexを指す。activeじゃない他のviewの入力$X_{i:i≠a}$は0 vectorとする。semantic spaceにおいてpivot viewのマッピングとそれ以外のviewのマッピングの相関の合計を最大化するような非線形マッピングを以下のようにして探す。

$$p = arg_{W_u} max_{W_1,...W_v} \sum_{j=1}^{N}{\frac{e^{αa cos(Yu,Ya,j)}}{􏰃\sum_{X'∈R^da}{e^{α cos(Yu,fa(X′,Wa))}}}}$$

我々のセットアップでは、pivot view$X_u$がuser featureであり、それ以外の付加的なviewが推薦したいそれぞれ異なるタイプのitem featureである。このようにパラメータをシェアすることで、あるドメインでデータが少なくても多のより多くあるドメインを通してよいmappingを学習することができる。

類似したnewsの嗜好を持つユーザは、他のモダリティにおいても類似した嗜好を持つという仮定が成立てば、この手法はうまくいくはずである。つまり、この仮定が正しければ、あらゆるドメインのサンプルが類似したユーザを全てのドメイン内でより正確にグルーピングすることを助ける。実験結果より、この仮定には合理性があることが確認できた。

### 5.1. Training MV-DNN
MV-DNNは確率的勾配降下法(Stochastic Gradient Decent; SGD)を用いて学習した。それぞれ訓練サンプルはuser viewとdata viewのペアの入力になっている。

### 5.2. Advantages of MV-DNN

利点というか従来のDSSMからの改良が2つある。

- 従来のDSSMはクエリviewと文書viewを同じサイズの特徴次元数で用いていた。でも実際は、全部同じサイズのでうまく表せない。例えば、tri-gramで全部表すとして、URLはwwwとかcomとかの接頭辞や接尾辞をもつ。これらが、同じ特徴としてマッピングされてしまう。でも我々はinputのraw textが短い時こういう問題は起こりにくいことを発見した。だから我々はその無駄な部分を削除して、category情報を入れた
- pair-wiseで学習することでscalabilityの向上を果たした

## 6. Dimension and data reduction

システムのスケーラビリティを向上させるために以下の４つを考えた。

### 6.1. Top Features
most frequent featureのTop-Kを選ぶ。前に述べたように、user raw featureをTF-IDFのスコアで前処理。

### 6.2. K-means
K-means clusteringを使う。


### 6.3. Local sensitive Hashing
Local sensitive hahing(LSH)を使う。

### 6.4. Reduce the Number of Training Examples
Train datasetのサンプル自体を減らす。ユーザに対応するそれぞれのviewのsampleがひとつになるように。各viewでユーザがライクしたすべてのアイテムのアベレージとる？

## 7. Experimental setup
MV-DNNを従来手法のMost Frequent, 一般的なSVD matrix decompositionを使ったCF, CCA(Top-K), Collaborative Topic Regression(CTR)と比較。

また、そのドメインで既にインタラクションがあるold userのデータセットと、そのドメインではインターネットはないけど、Bing上で検索とかブラウジングの履歴はあるnew userでも比較している。

データセットの作り方は、それぞれ全てのユーザを9:1でTrain setとTest setに分割。次にそれらを8:2でOld user setとNew user setに分割。Old userの方はitem viewの50%をTrainに利用、残りをTestに。New userの方は、Trainの方はitem viewなしでTestのみで利用。実験に用いたデータセットの概要はTable2。

![fig2](/Users/Corpy/Desktop/table2.png)

### 評価について
評価は、
それぞれ訓練セット内の$(user_i, item_j)$のペアに対して、9つの異なるアイテム$item_{r1},..., item_{r9}$をランダムに選ぶ。ここで$r1,..., r9$はランダムなindexを指す。そして、テスト用のペア$(user_i, item_{rk}), 1 \leq k \leq 9$を作り、Test setに加える。評価としては、どのくらい正しくシステムが正しいペア$(user_i, item_j)$を同じユーザの異なる他のアイテム$(user_i, item_{rk})$に対してよくrankできているかで評価。

評価指標は以下の2つ。

1. Mean Reciprocal Rank (MRR)平均逆順位  
推薦システムや情報検索の評価指標の一つ。
スコアをソートした結果、目的の情報（正解）がランキングされた順位を$r$とすると、単なる逆順位は、
$$RR = \frac{1}{R}$$
で表され、正解が1つ見つかればいいというような時に使う。値は0~1で、高いほどよい性能。例えば、正解が1番にランキングされていれば、$\frac{1}{1}=1$になり、10位にランキングされていれば、$\frac{1}{10}=0.11$になる。
平均逆順位は、正解が複数あるときにランキングの平均をとったもの。

$$MRR = \frac{1}{K} \sum_{i=1}^K{\frac{1}{r_i}}$$

2. Precision@1 (P@1)  
一番上に正解が来る確率


## 8. Result & discussion
Appsに対する推薦結果の評価はTable 3。Newsに対する結果はTable 4

MV-DNN圧勝！！ただ....MV-Top-K w/ Xboxとはなんぞや。と思ったら、X boxのMovie/TVのレコメはセンシティブだから出せないが、とりあえずこのviewも加えると、すごい他のviewのスコアも上がるよ。という話。

![fig2](/Users/Corpy/Desktop/table3.png)

![fig2](/Users/Corpy/Desktop/table4.png)

推薦結果の例は、Table 5

![fig2](/Users/Corpy/Desktop/table5.png)

## 9. Experiments of pablic data
パブリックデータで検証！！とかいいつつも、なぜかMV-DNNではなくSingle-view-DNNで比較。

![fig2](/Users/Corpy/Desktop/table6.png)


## 10. Algorithm scalability

Scalabilityの評価。CTRよりはいいですよー。

![fig2](/Users/Corpy/Desktop/table7.png)

## 11. Conclusion & future work
### Conclusion

- RecommendationのためのMulti-viewのモデル提案しました。各ドメインの共有の潜在表現を用いることで、そのドメインでログが少ないユーザにも推薦が出来ます。


>As a pilot study, we believe that **this work has opened a new door** to recommendation systems using deep learn- ing from multiple data sources.


### Future work

- もっと多くのuser featureをuser viewに使いたい
- 次元削減とかサンプル削ったりとかせずに、もっとscalableにしたい
- より多くのdomainを追加したい
- あとはどうやってCFと組み合わせるか考えたい

## 13. References
[1] Po-Sen Huang, Xiaodong He, Jianfeng Gao, Li Deng, Alex Acero, and Larry Heck. Learning deep structured semantic models for web search using clickthrough data. In CIKM’13, pages 2333–2338.


## おまけ

論文中に出てくるテクニカルタームの説明

### Unigram features
記号（単語や文字）列を統計分析する最も基本となるのは、それぞれの記号が文書中に現れる度数(頻度)である。さらに拡張した統計モデルは、2つの記号、3つの記号、…、n個の記号が隣接して出現する共起度数である。1つの記号、隣接する2つの記号、3つの記号、…n個の記号の度数を統計分析する方法をn-gram modelと呼ぶ。nが1つの記号の時がunigramである。

ここらへんは説明し始めると、色々説明せねばならず間違えそうなので、さわりだけ。一応前提として、言語モデルは以下。

### 言語モデル
単語が文書中に出現する過程を確率過程と見なし、ある単語がある位置に出現する確率はどれくらいかを計算するためのもの。自然言語処理における言語モデルの応用例としては、機械翻訳システムがある。機械翻訳システムでは生成した翻訳文がその適用先言語においてどれくらい尤もらしいかを言語モデルを使って定量的に評価することで、不自然な翻訳文が生成される可能性を減らすことができる。

I went to schoolという文書を考える。  

- unigram  
$P(I, went, to, school) = P(I)P(went)P(to)P(school)$  
確率は毎回独立

- bigram model
$P(I, went, to, school) = P(I)P(went|I)P(to|went)P(school|to)≠P(I, went, school, to)$  
確率は一つ前の目に依存

- n-gram model  
$P(w1,w2,…, wN) = ΠP(wk|wk-1, …, wk-n-1)$  
確率は1つ前からn-1つ前に依存

- skip-gram model  
n-gramでは連続した記号列だったが、skip-gramでは連続していなくてもいい

ココらへんは、以下の内田さんとダヌシカさんの資料がすごいわかりやすかった。読んでみるべし。

参考
- 内田将夫さん[6. 初歩の言語モデル](http://www2.nict.go.jp/univ-com/multi_trans/member/mutiyama/corpmt/6.pdf)
- Dr. Danushka Bollegala - [自然言語処理のための深層学習](http://cgi.csc.liv.ac.uk/~danushka/papers/DeepNLP.pdf)
- Dr. Danushka Bollegala - [大規模データから単語の意味表現学習-word2vec](http://www.slideshare.net/adtech-sat/word2vec20140413)
- 藤沼祥成さん - [言語モデル入門](http://www.slideshare.net/yoshinarifujinuma/04-12-labmeetingforpublic)
- 加藤誠さん - [情報検索モデル勉強会](http://www.slideshare.net/yukutakahashi/information-retrieval-model)

### TF-IDF
文書中の単語に関する重みの一種であり、主に情報検索や文章要約などの分野で利用される。いくつかの文書があったとき、それぞれの文書を特徴付ける単語がどれなのかTF-IDF値でわかる。

TFはTerm Frequencyで、それぞれの単語の文書内での出現頻度を表す。文書内に多く出てくる単語ほど重要という仮定に基づく。

$$tf(t,d) = \frac{n_{t,d}}{\sum_{s \in d}n_{s,d}}$$

$tf(t,d)$：文書$d$内のある単語$t$のTF値  
$n_{t,d}$：ある単語$t$の文書$d$内での出現回数  
$\sum_{s \in d}n_{s,d}$：文書$d$内のすべての単語の出現回数の和

IDFはInverse Document Frequency。Document Frequencyはれぞれの単語全ての文書内で何回現れたかを表し、その対数をとるのでIDF。非常に多くの文書で共通に使われている単語はそれほど重要ではないという仮定に基づく。

$$idf(t) = \log{\frac{N}{df(t)}} + 1$$

$idf(t)$：ある単語$t$のIDF値  
$N$：全文書数  
$df(t)$；ある単語$t$が出現する文書の数

対数をとっているのは、文書数の規模にに対して、IDFの値の変化を小さくするため。この2つの値を掛けたものをそれぞれの単語の重みにすれば、その値が大きいほど各文書を特徴付ける単語になる。

参考
- [特徴抽出とTF-IDF](http://qiita.com/ynakayama/items/300460aa718363abc85c)
- [TF-IDFで文書内の単語の重み付け](http://takuti.me/note/tf-idf/)
