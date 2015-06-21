#A Multi-View Deep Learning Approach for Cross Domain User Modeling in Recommendation System

## Ali Elkahky, Yang Song, Xiaodong He,  WWW'15 [PDF](http://research.microsoft.com/pubs/238334/frp1159-songA.pdf)

## Abstract

- 近年、オンラインサービスは大量のユーザに対して、関連のあるコンテンツを推薦するために，personalizeする事が大事
-  その時に必要となるのは、コールドスタート問題（新規ユーザ）に対応すること
- この研究では、推薦の質とスケーラビリティの両方を実現する内容ベース推薦システムを提案する
- ユーザのウェブ上でのブラウジング履歴や検索クエリを用いて、ユーザをリッチに表現する
- ユーザとアイテムを、ユーザとユーザが好むアイテム間の関連性を最大化するような潜在空間にマップするためにDeep learningを用いた
- Multi-view deep learning modelを導入することで、異なるドメインのアイテム特徴とユーザ特徴から一緒に学習できる
- 我々は、入力特徴の次元とトレーニングデータの量を削減することによってユーザ表現に基づくリッチな特徴をスケーラブルに作成する方法を提案した
- リッチなユーザ特徴表現は、関連のあるユーザの行動パターンを学習し，有用な推薦を（まだ何もサービスとインタラクションがない新規の）ユーザに与えることを可能にする．
- 異なるドメインを組み合わせて学習を単一のモデルにすることは、ドメインを超えた全てのドメインにおいて、よりコンパクトでセマンティック的に豊かなユーザ潜在特徴ベクトルを作成可能なことと、推薦の質の向上を導く
- 実際に以下の3つのレコメンデーションシステムで評価実験した 
 1. Windows Apps recommendation
 2. News recommendation
 3. Movie/TV recommendation
- 実験により，提案手法がstate-of-the-art algorithmより優れていることを確認した（既存ユーザに対して49%向上, 新規ユーザに対して115%向上）
- パブリックに公開されているデータセットでも実験を行い、クロスドメインのレコメンデーションシステムを構成する上で、トラディショナルなトピックモデルの手法より優れていることを確認した
- Scalabilityは、提案手法は数100万規模のユーザと数10億規模のアイテムに対して容易にスケールできる
- 全てのドメインから得られる特徴を組み合わせることは、それぞれのドメインにモデルを分けてやるより良いパフォーマンスであることを実験で確認した

## Gneral Terms
User Modeling

## Key words
User Modeling, Recommendation System, Multi-view learning, Deep learning

## Introduction

- 推薦システムとpersonalizationのweb serviceでの重要性は増してきている
- ユーザと相関の高いアイテムを出来る限り早く見つけたい
- 主要なアプローチ
	- 協調フィルタリング(Collaborative filtering: CF)
		- ユーザのアイテム閲覧履歴(や評価)の類似性をもとに推薦
	- 内容ベースフィルタリング(Content-based filtering: CB)
		- ユーザやアイテムの特徴の類似性をもとに推薦
- 両方アプリケーションとして一定の成果上げてる
- Challengesはpersonalizationと推薦の質
- CFは履歴をもとに推薦するから、質の高い推薦するためにはユーザのアイテムへの閲覧履歴がたくさん必要になる。この問題は**cold start problem**として知られている
- 新規のweb serviceでは履歴がないのでこのcold start problemがより顕著。よって、トラディショナルなCFは新規ユーザに対してはいい推薦できない
- 逆に、CBは推薦するためにユーザやアイテムの特徴を使う。例えば、新しい記事$N_i, N_j$が同じトピックを共有しており、ユーザが$N_i$を好きな場合、システムは$N_j$をユーザに推薦できる。同様に、ユーザ$U_i, U_j$が住んでいる場所や年齢、性別などが類似している場合、システムは$U_i$が過去にlikeしたものを$U_j$に推薦できる（これは履歴を使っているけど、CFと異なるのは、ユーザの特徴の類似性から推薦しているとこ。CFはユーザの履歴の類似性から推薦する）
- 実際に研究でも、CBはcold start problemに対応可能だと示されている
- **しかし実際はその有効性には疑問がある。なぜなら、ユーザの特徴を限られた情報から獲得することは一般に難しく、だいたいは実際のユーザの興味を正確に捉えることはできない**
- このような限界に挑戦するために、我々は、ユーザとアイテムの両方をうまく利用した推薦システムを提案する
- ユーザのprofileを用いるのではなく、ユーザのブラウジングやサーチの履歴からユーザの興味をモデル化して、リッチなユーザ表現を獲得する手法を提案する
- 根本的な仮定として、ユーザのオンライン上での行動はユーザのバックグラウンドや嗜好を反映するので、ユーザが興味があるであろうアイテムやトピックの正確な洞察を得ることができる
- 例えば、幼児に関連する検索クエリやサイト(トイザらスなど)の訪問は、彼女が幼児の母であることを提案できるだろう
- 潤沢にあるユーザのオンライン上の行動を用いることで、より効率的かつ効果的な推薦ができる
- 我々の研究では、Deep Structured Semantic Models (DSSM)を拡張して、ユーザとアイテムを共通のsemantic spaceにマップし、そのマップした潜在空間上で最もそのユーザと相関が高いアイテムを推薦する
- 我々のモデルでは、ユーザとアイテムを、非線形変換を繰り返すことで、コンパクトなshare latent semantic space（射影後のユーザと、射影後の(ユーザが興味を示した)アイテムの相関が最大になるような空間）に射影する
- 例えば、fifa.comに訪れた人はワールドカップのニュースを読むことや、Xboxでサッカーゲームをすることを好むであろうというようなマッピングを学習する
- ユーザ側の豊かな特徴は、ユーザの行動をモデリングすることを可能にするので、従来のCBの多くの制限に打ち勝つことができる
- もちろんユーザのcold start problemにも対応可能
- たとえば、ユーザが音楽サービス上で何の履歴もなかったとしても、我々のモデルによってユーザの興味が得られていれば、音楽に関連するアイテムを推薦できる
- 我々のdeep learningモデルは、ranking-basedの目的をもっており、これは推薦システムに有効である
- さらに、single-view のDNNであるオリジナルのDSSMモデルを異なるドメインのアイテムから同時に学習するMulti-viewモデル（MV-DNN）に拡張した
- 共有の特徴空間を共有せずに学習するmulti-view learningはよく研究されている
- 我々はMV-DNNをmulti-view learningにおける一般的なdeep learningのアプローチとして考えた
- 特に、データセット（news, app, movie/tv logs）を用いて、各々のドメイン内でユーザ特徴からアイテム特徴にマップするseparateなモデルを構築する代わりに、全てのドメインのアイテムの特徴量を共通に最適化するmulti-view modelを構築した
- MV-DNNは全てのドメインのユーザ嗜好データを用いているので、ドメインを超えて有効なユーザ表現を学習できる上、data sparsity problemにも対応することができる 
- MV-DNNモデルは実験により、全てのドメインにおいて推薦の質を向上させたことを確認した
- さらに、非線形マッピングにより、コンパクトなユーザ表現を得ることができ、これは、様々なタスクに使える
- Deep learningを使ってリッチなユーザ表現を得ることのChallengeは特徴空間が高次元になってしまい学習が難しくなること
- 本研究では、scalabilityを実現するため、様々な次元削減技術を提案している
- コントリビューションは以下の5つ
1. recommendation systemを構築するため、リッチなユーザ特徴を用いた
2. deep learningを用いたcontent-based recommendation systemを提案し、scalableなシステムを実現するための様々なテクニックを調査した
3. 複数のドメインを組み合わせるMulti-view deep learningモデルを導入し、recommendation systemを構築した
4. MV-DNNから学習されたsemantic feature mappingによりユーザのcold start problemに対応した
5. 実世界のビッグデータを用いた厳密な評価実験により、state-of-the-artを大きく上回る有効性を確認した

## Related work

- 

There has been extensive study on recommendation sys- tems with a myriad of publications. In this section, we aim at reviewing a representative set of approaches that are mostly related to our proposed approach.

In general, recommendation systems can be divided in- to collaborative recommendation and content based recom- mendation. Collaborative Recommendation systems recom- mend an item to a user if similar users liked this item. Exam- ples of this technique include nearest neighbor modeling [3], Matrix Completion [19], Restricted Boltzmann machine [22], Bayesian matrix factorization [21] etc. Essentially, these ap- proaches are user collaborative filtering, item collaborative filtering or both item and user collaborative filtering. In user collaborative filtering such as [3], the algorithm com- putes the similarity between users based on items they liked. Then, the scores of user-item pairs are computed by combin- ing scores of this item given by similar users. Item based col- laborative filtering [23], computes similarity between items based on users who like both items, then recommend the user items similar to the ones she liked before. User-itembased collaborative filtering finds a common space for items and users based on user-item matrix and combines the item and user representation to find a recommendation. All ma- trix factorization approaches like [19] and [21] are examples of this technique. CF can be extended to large-scale setups like in [6]. However, CF is generally unable to handle new users and new items, a problem which is often referred to as cold-start issue.
The second approach for recommendation systems is content- based recommendation. This approach extracts features from item’s and/or user’s profile and recommend items to users according to these features. The underlying assump- tion is that similar users tend to like items similar to the items they liked previously. In [14], a method is proposed to construct a search query with some features of items the user liked before to find other relevant items to recommend. An- other example is presented in [15] where each user is modeled by a distribution over News topics that is constructed from articles she liked with a prior distribution of topic preference computed using all users who share the same location. This approach can handle new items (News articles) but for new users the system used location feature only which implies that new users are expected to see most frequent topics in their location. This might be a good features to recommend News but in other domains, for example Apps recommen- dation, using only location information may not work as a good prior over user’s preferences.
Recently, researchers have developed approaches that com- bine both collaborative recommendation and content based recommendation. In [16], the author used item features to smooth user data before using collaborative filtering. In [7], the authors used Restricted Boltzmann Machine to learn similarity between items, and then combined this with col- laborative filtering. A Bayesian approach was developed in [32] to jointly learn the distribution of items, research pa- pers in their case, over different components (topics) and the factorization of the rating matrix.
Handling the cold start issue in recommendation systems is studied mainly for new items (items that have no rating by any user). As we mentioned before, all content based filtering can handle cold start for item, and there are some methods that were developed and evaluated specifically for this issue like in [24] and [7]. The work in [18] studied how to learn user preferences for new users incrementally by rec- ommending items that give the most information about user preferences while minimizing the probability of recommend- ing irrelevant content. User modeling via rich features have been studied a lot recently. For example, it has been shown that user search queries can be used to discover the similari- ties between users [25]. Rich features from user search histo- ry has also been used for personalized web search [26]. For recommendation systems, the authors in [2] leveraged the user’s historical search queries to build personalized taxono- my for recommending Ads. On the other hand, researchers have discovered that a user’s social behaviors can also be used to build the profile of the user. In [1], the authors used user’s tweets in Twitter data to recommend News articles.
Most traditional recommendation system research focused on data within a single domain. Recently, there has been an increasing interest in cross domain recommendation. There are different approaches for addressing cross domain rec- ommendation. One approach is to assume that different domains share similar set of users but not the items, as il-lustrated in [20]. In their work, the authors augmented data from rating of movies and books from datasets that have common users. The augmented data set was then used to perform collaborative filtering. They showed that this in particular helped the cases where users with little profile information in one of the domains (cold-start users). The second approach addressed the scenarios where the same set of items shared different types of feedbacks in different domains like user clicks or user explicit rating. As shown in [17], the authors introduced a coordinate system trans- fer method for cross domain matrix factorization. In [12], the authors studied the cross domain recommendation in the case where there existed no shared users or items be- tween domains. They developed a generative model to dis- cover common clusters between different domains. However, a challenge in their approach is its ability to scale beyond medium datasets due to the computational cost. A different approach was introduce in [28] for author collaboration rec- ommendation where they built a topic model to recommend authors to collaborate from different research fields.

For many approaches in recommendation systems the ob- jective function is to minimize the root mean squared error on the user-item matrix reconstruction. Recently, ranking based objective function has shown to be more effective in giving better recommendation as shown in [11].
Deep learning has recently been proposed for building rec- ommendation systems for both collaborative and content based approaches. In [22], an RBM model was used for collaborative filtering. Deep learning for content based rec- ommendation has been done for example in [30] where deep learning was applied to learn embedding for music features. This embedding was then used to regularize matrix factor- ization in collaborative filtering.

## Description of the data sets

- 以下の4つのデータセットを用いた
1. Bingから得られる検索ログ
2. Bing Newsから得られるニュース記事のブラウジングログ
3. Windows AppStoreから得られるAppのダウンロードログ
4. Xboxから得られるMovie/TVの視聴ログ
- すべてのログは、英語を公用語とするマーケット（US, Canada, UK）で2013.12-2014.6の間に収集された

ユーザ特徴
- Bing上でのユーザの検索クエリとクリックしたURLを収集
- クエリはまず正規化され、unigram features（つまり一つ一つの独立した単語特徴）に分割した。URLは次元削減のためdomain-levelに短縮した（例えば、www.linkedin.comなど）
- 最も人気で価値のある特徴だけを保つために、TF-IDFのスコアを用いた
- 全部で、300万のunigram featuresと50万のdomain featuresを獲得し、ユーザ特徴として、合計350万の特徴ベクトルを得た

Overall, we selected 3 million unigram features and 500K domain features, leading to a total length of 3.5-million user feature vector.
(News Features) We collected news article clicks from Bing News vertical. Each News item is represented by three parts of features. The first part is the title features encoded using letter tri-gram representation as we will describe in the next section. Secondly, the top-level category of each News (e.g., Entertainment) is encoded as binary features. Finally, the Named Entities in each article, extracted using



## 参考資料

論文中に出てくるテクニカルタームの説明

### Unigram features
記号（単語や文字）列を統計分析する最も基本となるのは、それぞれの記号が文書中に現れる度数(頻度)である。さらに拡張した統計モデルは、2つの記号、3つの記号、…、n個の記号が隣接して出現する共起度数である。1つの記号、隣接する2つの記号、3つの記号、…n個の記号の度数を統計分析する方法をn-gram modelと呼ぶ。nが1つの記号の時がunigramである。

ここらへんは説明し始めると、色々説明せねばならず間違えそうなので、さわりだけ。一応前提として、言語モデルは以下。

### 言語モデル
単語が文書中に出現する過程を確率過程と見なし、ある単語がある位置に出現する確率はどれくらいかを計算するためのもの。自然言語処理における言語モデルの応用例としては、機械翻訳システムがある。機械翻訳システムでは生成した翻訳文がその適用先言語においてどれくらい尤もらしいかを言語モデルを使って定量的に評価することで、不自然な翻訳文が生成される可能性を減らすことができる。

I went to schoolという文書を考える
- unigram  
$P(I, went, to, school) = P(I)P(went)P(to)P(school)$
確率は毎回独立

- bigram model
$P(I, went, to, school) = P(I)P(went|I)P(to|went)P(school|to)≠P(I, went, school, to)$
確率は一つ前の目に依存

- n-gram model
$P(w1,w2,…, wN) ~ ΠP(wk|wk-1, …, wk-n-1)$
確率は1つ前からn-1つ前に依存

- skip-gram model


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

