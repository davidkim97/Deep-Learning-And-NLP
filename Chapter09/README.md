# Chapter 09. 워드 임베딩(Word Embedding)

### 09-1 워드 임베딩(Word Embedding)

- 워드 임베딩(Word Embedding)은 단어를 벡터로 표현하는 방법.

- 단어를 밀집 표현으로 변환.

- 희소 표현(Sparse Representation)

    - 벡터 또는 행렬(matrix)의 값이 대부분이 0으로 표현되는 방법을 희소 표현 이라고 한다.

    - 원-핫 인코딩을 통해 나온 원-핫 벡터는 희소 벡터(sparse vector)이다.

    - 희소 벡터는 단어의 개수가 늘어나면 벡터의 차원이 한없이 커진다.

    - 이러한 벡터 표현은 공간적 낭비를 불러일으킨다.

    - 원-핫 벡터와 같이 희소 벡터의 문제점은 단어의 의미를 표현하지 못한다.

- 밀집 표현(Dense Representation)

    - 밀집 표현은 벡터의 차원을 단어 집합의 크기로 상정하지 않는다.

    - 사용자가 설정한 값으로 모든 단어의 벡터 표현의 차원을 맞춘다.


- 워드 임베딩(Word Embedding)

    - 단어를 밀집 벡터(dense vector)의 형태로 표현하는 방법을 워드 임베딩(word embedding)이라고 한다.

    - 이 밀집 벡터를 워드 임베딩 과정을 통해 나온 결과라고 하여 임베딩 벡터(embedding vector)라고도 한다.

    ||원-핫 벡터|임베딩 벡터|
    |---|---|---|
    |차원|고차원(단어 집합의 크기)|저차원|
    |다른 표현|희소 벡터의 일종|밀집 벡터의 일종|
    |표현 방법|수동|훈련 데이터로부터 학습함|
    |값의 타입|1과 0|실수|

---
### 09-2 워드투벡터(Word2Vec)

- 단어 벡터 간 유의미한 유사도를 반영할 수 있도록 단어의 의미를 수치화 하는 방법.

- 희소 표현(Sparse Representation)

    - 원-핫 인코딩을 통해 얻은 원-핫 벡터는 표현하고자 하는 단어의 인덱스의 값만 1이고, 나머지 인덱스에는 전부 0으로 표현되는 벡터 표현 방법

    - 벡터 또는 행렬의 값이 대부분이 0으로 표현되는 방법을 희소 표현(sparse represntation)이라고 한다.

    - 이러한 표현 방법은 각 단어 벡터간 유의미한 유사성을 표현할 수 없다는 단점이 있고, 대안으로 단어의 의미를 다차원 공간에 벡터화하는 방법을 사용하는데 이러한 표현을 분산 표현(distributed representation)라고 한다.

    - 분산 표현을 이용하여 단어 간 의미적 유사성을 벡터화하는 작업을 워드 임베딩이라고 한다.

- 분산 표현(distributed representation)

    - 분산 표현(distributed representation)방법은 기본적으로 분포 가설(distributional hypothesis)이라는 가정 하에 만들어진 표현 방법.

    - 가정은 <span style='font-weight:bold; color:tomato'>'비슷한 문맥에서 등장하는 단어들은 비슷한 의미를 가진다'</span>라는 가정.

    - 분산 표현은 분포 가설을 이용하여 텍스트를 학습하고, 단어의 의미를 벡터의 여러 차원에 분산하여 표현.

    - 분산 표현은 저차원에 <span style='font-weight:bold; color:tomato'>단어의 의미를 여러 차원에다가 분산 </span>하여 표현

    - 이런 표현 방법을 사용하면 <span style='font-weight:bold; color:tomato'>단어 벡터 간 유의미한 유사도</span>를 계산할 수 있음.

- CBOW(Continuous Bag of Words)

    - Word2Vec의 학습 방식에는 CBOW(Continuous Bag of Words)와 Skip-Gram 두 가지 방식이 있음.

        - CBOW는 주변에 있는 단어들을 입력으로 중간에 있는 단어들을 예측하는 방법.

        - Skip-Gram은 중간에 있는 단어들을 입력으로 주변 단어들을 예측하는 방법.

    - 