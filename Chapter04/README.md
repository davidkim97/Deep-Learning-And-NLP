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