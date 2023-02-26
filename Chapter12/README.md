# Chapter 12. 태깅 작업(Tagging Task)

- 단어의 유형이 사람, 장소, 단체 등 어떤 유형인지를 알아내는 개체명 인식(Named Entity Recognitaion)

- 단어의 품사가 명사, 동사, 형용사 인지를 알아내는 품사 태깅(Part-of Speech Tagging)

### 12-1 케라스를 이용한 태깅 작업 개요(Tagging Task using Keras)

- 태깅 작업은 지도 학습(Supervised Learning)에 속함.

- 태깅해야 하는 단어 데이터를 X, 레이블에 해당되는 태깅 정보 데이터는 y

- X에 대한 훈련 데이터는 X_train, 테스트 데이터는 X_test, y에 대한 훈련데이터는 y_train, 테스트 데이터는 y_test

| #   | X_train                                                                 | y_train                                                | 길이 |
| --- | ----------------------------------------------------------------------- | ------------------------------------------------------ | ---- |
| 0   | ['EU', 'rejects', 'German', 'call', 'to', 'boycott', 'British', 'lamb'] | ['B-ORG', 'O', 'B-MISC', 'O', 'O', 'O', 'B-MISC', 'O'] | 8    |
| 1   | ['peter', 'blackburn']                                                  | ['B-PER', 'I-PER']                                     | 2    |
| 2   | ['brussels', '1996-08-22' ]                                             | ['B-LOC', 'O']                                         | 2    |
| 3   | ['The', 'European', 'Commission']                                       | ['O', 'B-ORG', 'I-ORG']                                | 3    |

- 시퀀스 레이블링(Sequence Labeling)

  - 위 같은 입력 시퀀스 X = [x1, x2, x3....xn]에 대하여 레이블 시퀀스 y = [y1, y2, y3...yn]를 각각 부여하는 작업을 시퀀스 레이블링 작업(Sequence Labeling Task)라고 함.

- 양방향 LSTM(Bidirectional LSTM)

  ```python
  model.add(Bidirectional(LSTM(hidden_units, return_sequences = True)))
  ```

  - LSTM을 사용하는 이유는 이전 시점의 단어 정보 뿐만 아니라, 다음 시점의 단어 정보도 참고하기 위함.

  - 양방향은 기존의 단방향 LSTM()을 Bidirectional() 안에 넣으면 됨.

---

### 12-02 양방향 LSTM를 이용한 품사 태깅(Part-of-speech Tagging using Bi-LSTM)
