import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 모델 파일 로드
model = load_model('model.h5')

# 토크나이저 초기화
tokenizer = Tokenizer()

max_len = 100  # 최대 시퀀스 길이 설정

test_sentence = input("하고싶은 말을 하세요: ")
test_sentence = test_sentence.split(' ')
test_sentences = []
now_sentence = []
for word in test_sentence:
    now_sentence.append(word)
    test_sentences.append(now_sentence[:])

test_X_1 = tokenizer.texts_to_sequences(test_sentences)
test_X_1 = pad_sequences(test_X_1, padding='post', maxlen=max_len)
prediction = model.predict(test_X_1)

result_emotion = {0: "중립", 1: "혐오", 2: "공포", 3: "행복", 4: "분노", 5: "슬픔", 6: "놀람"}

for idx, sentence in enumerate(test_sentences):
    print(sentence)
    print(prediction[idx])
    result = prediction[idx]
    res_ind = np.argmax(result)
    print("감정분석 결과:", result_emotion[res_ind])
