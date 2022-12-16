# Chapter 04. 카운트 기반의 단어 표현(Count based word Representation)

### 04-1 다양한 단어의 표현 방법

- 단어의 표현 방법

    - 단어의 표현 방법은 크게 국소 표현(Local Representation)방법과 분산 표현(Distributed Representation)방법으로 나뉜다.

    - 국소 표현 방법은 해당 단어 그 자체만 보고, 특정값을 맵핑하여 단어를 표현하는 방법

    - 분산 표현 방법은 그 단어를 표현하고자 주변을 참고하여 단어를 표현하는 방법

- 단어 표현의 카테고리화

    - Bag of Words는 국소 표현(Local Representation)에 속하며, 단어의 빈도수를 카운트(Count)하여 단어를 수치화하는 단어 표현 방법.

---
### 04-2 Bag of Words(BoW)

- Bag of Words란?
    - 단어들의 순서는 전혀 고려하지 않고, 단어들의 출현 빈도(frequency)에만 집중하는 텍스트 데이터의 수치화 표현 방법.

    - BoW를 만드는 과정을 쉽게 표현하면
    ```
    (1) 각 단어에 고유한 정수 인덱스를 부여 -> 단어 집합 생성.
    (2) 각 인덱스의 위치에 단어 토큰의 등장 횟수를 기록한 벡터를 만든다.
    ```

    - 한국어 예제를 통해 BoW 이해하기
    ```python
    from konlpy.tag import Okt

    okt = Okt()

    def build_bag_of_words(document):
    # 온점 제거 및 형태소 분석
    document = document.replace('.', '')
    tokenized_document = okt.morphs(document)

    word_to_index = {}
    bow = []

    for word in tokenized_document:  
        if word not in word_to_index.keys():
        word_to_index[word] = len(word_to_index)  
        # BoW에 전부 기본값 1을 넣는다.
        bow.insert(len(word_to_index) - 1, 1)
        else:
        # 재등장하는 단어의 인덱스
        index = word_to_index.get(word)
        # 재등장한 단어는 해당하는 인덱스의 위치에 1을 더한다.
        bow[index] = bow[index] + 1

    return word_to_index, bow
    ```

- CountVectorizer 클래스로 BoW 만들기

    - 사이킷 런에서는 단어의 빈도를 Count하여 Vector로 만드는 CountVectorizer 클래스를 지원

    ```python
    from sklearn.feature_extraction.text import CountVectorizer

    corpus = ['you know I want your love. because I love you.']
    vector = CountVectorizer()

    # 코퍼스로부터 각 단어의 빈도수를 기록
    print('bag of words vector :', vector.fit_transform(corpus).toarray()) 

    # 각 단어의 인덱스가 어떻게 부여되었는지를 출력
    print('vocabulary :',vector.vocabulary_)
    ```

    - CountVectorizer는 기본적으로 길이가 2이상인 문자에 대해서만 토큰으로 인식한다.

    - CountVectorizer는 단지 띄어쓰기만을 기준으로 단어를 자르는 낮은 수준의 토큰화를 진행하고 BoW를 만든다.

    - 영어의 경우 띄어쓰기만으로 토큰화가 수행되기 때문에 문제가 없지만 한국어에 CountVectorizer를 적용하면, 조사등의 이유로 제대로 BoW가 만들어 지지 않음

- 불용어를 제거한 BoW 만들기

    - BoW를 만들때 불용어를 제거하는 일은 자연어 처리의 정확도를 높이기 위해서 선택할 수 있는 전처리 방법이다.

    - 영어의 BoW를 만들기 위해 사용하는 CountVectorizer는 불용어를 지정하면, 불용어는 제외하고 BoW를 만들 수 있도록 불용어 제거 기능을 지원
    ```python
    from sklearn.feature_extraction.text import CountVectorizer
    from nltk.corpus import stopwords
    ```

---
### 04-3 문서 단어 행렬(Document-Term Matrix, DTM)

- 문서 단어 행렬(Document-Term Matrix, DTM)의 표기법

    - 다수의 문서에서 등장하는 각 단어들의 빈도를 행렬로 표현한 것을 말한다.

    - 예를 들어 표현
        ```
        문서1 : 먹고 싶은 사과
        문석2 : 먹고 싶은 바나나
        문서3 : 길고 노란 바나나 반나
        문서4 : 저는 과일이 좋아요
        ```

        ||과일이|길고|노란|먹고|바나나|사과|싶은|저는|좋아요|
        |---|---|---|---|---|---|---|---|---|---|
        |문서1|0|0|0|1|0|1|1|0|0|
        |문서2|0|0|0|1|1|0|1|0|0|
        |문서3|0|1|1|0|2|0|0|0|0|
        |문서4|1|0|0|0|0|0|0|1|1|

        - 문서에서 등장한 단어의 빈도를 행렬의 값으로 표기.

        - 문서 단어 행렬은 문서들을 서로 비교할 수 있도록 수치화할 수 있다는 점에서 의의를 갖는다.

        - 형태소 분석기로 단어 토큰화를 수행하고, 불용어에 해당되는 조사들 또한 제거하여 더 정제된 DTM을 만들 수도 있을 것이다.

- 문서 단어 행렬(Document-Term Matrix)의 한계
    
    - 희소 표현(Sparse representation)

        - 가지고 있는 전체 코퍼스가 방대한 데이터라면 문서 벡터의 차원은 수만 이상의 차원을 가질 수 있다.

        - 많은 문서 벡터가 대부분의 값이 0을 가질 수도 있다.

        - 원-핫 벡터나 DTM과 같은 대부분의 값이 0인 표현을 희소 벡터(sparse vector)또는 희소 행렬(sparse matrix)라고 부르는데,  
        희소 벡터는 많은 양의 저장 공간과 높은 계산 복잡도를 요구한다.

    - 단순 빈도 수 기반 접근

        - 여러 문서에 등장하는 모든 단어에 대해서 빈도 표기를 하는 이런 방법은 때로는 한계를 가지기도 한다.

---
### 04-4 TF-IDF(Term Frequency-Inverse Document Frequency)

- TF-IDF(단어 빈도-역 문서 빈도, Term Frequency-Inverse Document Frequency)
    
    - 단어의 빈도와 역 문서 빈도(문서의 빈도에 특정 식을 취함)를 사용하여 DTM 내의 각 단어들마다 중요한 정도를 가중치로 주는 방법.

    - 어떤 단어가 하나의 문서에도 맣이 사용되었다고 하더라도, 다른 모든 문서에서 널리 쓰이는 흔해 빠진 단어라면 이 단어는 특정성(specificity)이 떨어지는 것이다.

    - TF-IDF는 주로 문서의 유사도를 구분하는 작업, 검색 시스템에서 검색 결과의 중요도를 정하는 작업, 문서 내에서 특정 단어의 중요도를 구하는 작업 등에 쓰인다.

- 파이썬으로 TF-IDF 직접 구현하기
    ```python
    import pandas as pd
    from math import log # IDF 계산을 위해

    docs = [
    '먹고 싶은 사과',
    '먹고 싶은 바나나',
    '길고 노란 바나나 바나나',
    '저는 과일이 좋아요'
    ] 
    vocab = list(set(w for doc in docs for w in doc.split()))
    vocab.sort()

    # TF, IDF, TF-IDF 값을 구하는 함수

    # 총 문서의 수
    N = len(docs) 

    def tf(t, d):
    return d.count(t)

    def idf(t):
    df = 0
    for doc in docs:
        df += t in doc
    return log(N/(df+1))

    def tfidf(t, d):
    return tf(t,d)* idf(t)

    # TF 구해보기, 다시 말해 DTM을 데이터프레임에 저장하여 출력
    result = []

    # 각 문서에 대해서 아래 연산을 반복
    for i in range(N):
    result.append([])
    d = docs[i]
    for j in range(len(vocab)):
        t = vocab[j]
        result[-1].append(tf(t, d))

    tf_ = pd.DataFrame(result, columns = vocab)

    tf_

    # 각 단어에 대한 IDF 값 구하기

    result = []
    for j in range(len(vocab)) :
        t = vocab[j]
        result.append(idf(t))

    idf_ = pd.DataFrame(result, index = vocab, columns = ["IDF"])
    idf_

    # TF-IDF 행렬을 출력
    result = []
    for i in range(N) :
        result.append([])
        d = docs[i]
        for j in range(len(vocab)) :
            t = vocab[j]
            result[-1].append(tfidf(t,d))

    tfidf_ = pd.DataFrame(result, columns = vocab)
    tfidf_
    ```

- 사이킷런을 이용한 DTM, TF-IDF 실습

    - 사이킷런을 통해 DTM과 TF-IDF를 만들 수 있는데, CountVectorizer를 사용하면 DTM을 만들 수 있다.

    