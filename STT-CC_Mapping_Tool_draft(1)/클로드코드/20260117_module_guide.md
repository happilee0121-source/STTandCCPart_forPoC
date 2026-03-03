# 고급 전처리 & NLP 모듈 사용 가이드

## 📋 개요

자연어 음성 회의록을 처리하기 위한 고급 전처리 및 NLP 모듈입니다.

### 주요 기능

**Layer 2: Advanced Pre-processing**
- ✅ 잡음 제거 (필러 단어, 반복, 배경 소음)
- ✅ 정규화 (맞춤법, 띄어쓰기, 문장 경계)
- ✅ 파편화된 발화 병합
- ✅ 화자 이름 정규화

**Layer 3: Advanced Semantic NLP**
- ✅ 한국어 형태소 분석
- ✅ 다층 의도 분류 (7가지 의도 타입)
- ✅ 감정 분석 (positive/neutral/negative)
- ✅ 핵심 키워드 추출

## 🚀 빠른 시작

### 1. 기본 사용법

```python
import yaml
from src.preprocessing.advanced_preprocessor import AdvancedPreprocessor
from src.nlp.advanced_intent_classifier import AdvancedIntentClassifier

# 설정 로드
with open('config/config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

# 전처리
preprocessor = AdvancedPreprocessor(config)
utterances = preprocessor.process(
    input_path='data/input/your_meeting.txt',
    output_path='data/output/utterances.json',
    enable_merge=False  # 파편 병합 여부
)

# NLP 분석
classifier = AdvancedIntentClassifier(config)
intents = classifier.process(
    utterances_path='data/output/utterances.json',
    output_path='data/output/intents.json'
)
```

### 2. 통합 테스트 실행

```bash
python test_advanced_modules.py
```

## 📝 입력 데이터 형식

### 자연스러운 음성 회의록 (권장)

```
[00:00:05] 김팀장: 음... 네, 그러면 오늘 오늘 보안 요구사항에 대해서 논의를 시작하도록 하겠습니다.

[00:00:18] 박대리: 네네, 먼저 먼저 사용자 인증 부분부터 얘기를 해볼까요? 현재 패스워드 정책이 좀 너무 약한 것 같아요. 그 뭐...

[00:00:35] 김팀장: 맞습니다. 최소한 8자리는 넘어야 하고, 그 영문하고 숫자, 특수문자 조합이 반드시 필요할 것 같습니다.
```

### 특징
- 필러 단어 포함 (`음...`, `그 뭐`, `약간`)
- 반복 (`오늘 오늘`, `네네`, `먼저 먼저`)
- 불완전 문장 (`그 뭐...`)
- 배경 소음 표시 가능 (`(웃음)`, `(박수)`)

## 🔧 Layer 2: 고급 전처리

### 잡음 제거 기능

**1. 필러 단어 제거**
```
입력: "음... 네, 먼저 먼저 사용자 인증 부분부터..."
출력: "네, 사용자 인증 부분부터..."
```

제거 대상 필러:
- 음, 어, 아, 그, 뭐, 저
- 이제, 막, 약간, 좀
- 진짜, 그냥, 뭔가, 그러니까

**2. 반복 제거**
```
입력: "오늘 오늘 보안 요구사항"
출력: "오늘 보안 요구사항"

입력: "네네네 좋습니다"
출력: "네 좋습니다"
```

**3. 배경 소음 제거**
```
입력: "좋은 의견입니다 (웃음) 그리고..."
출력: "좋은 의견입니다 그리고..."
```

### 정규화 기능

**1. 문장 부호 정규화**
```
입력: "맞습니다.최소한 8자리는"
출력: "맞습니다 . 최소한 8자리는"
```

**2. 따옴표 통일**
```
입력: "AES-256" 사용
출력: "AES-256" 사용
```

**3. 연속 공백 제거**
```
입력: "보안    요구사항"
출력: "보안 요구사항"
```

### 파편 병합 기능

불완전한 문장들을 자동으로 병합:

```python
# enable_merge=True 설정 시
입력 발화 1: "데이터 암호화를..."
입력 발화 2: "AES-256 알고리즘을 사용하면 됩니다."
출력: "데이터 암호화를 AES-256 알고리즘을 사용하면 됩니다."
```

종결 어미 패턴:
- 다. / 요. / 까. / 죠. / 네.
- 까? / 요? / 나?
- 다! / 요!

### 화자 정규화

```
입력: "발표자1(김철수)"
출력: "발표자1"

입력: "발표자  1"
출력: "발표자1"
```

## 🧠 Layer 3: 고급 NLP 분석

### 의도 분류 (7가지)

**1. requirement (요구사항)**
```
"최소 8자리 이상의 패스워드가 필요합니다"
→ requirement (mandatory)
```

**2. question (질문)**
```
"접근 제어는 어떻게 할까요?"
→ question (method)
```

**3. agreement (동의)**
```
"네, 좋습니다"
→ agreement
```

**4. disagreement (반대)**
```
"아니요, 그건 문제가 있습니다"
→ disagreement
```

**5. suggestion (제안)**
```
"RBAC을 도입하면 어떨까요?"
→ suggestion
```

**6. clarification (명확화)**
```
"즉, 역할 기반 접근 제어를 말씀하시는 거죠"
→ clarification
```

**7. objection (이의)**
```
"하지만 그건 고려해야 할 사항이 있습니다"
→ objection
```

### 세부 의도 분류

**requirement 세부 분류:**
- `mandatory`: 필수 요구사항 ("반드시", "꼭")
- `recommended`: 권장 요구사항 ("권장", "추천")
- `general`: 일반 요구사항

**question 세부 분류:**
- `feasibility`: 가능성 질문 ("가능한가요?")
- `method`: 방법 질문 ("어떻게?")
- `general`: 일반 질문

### 감정 분석

**positive (긍정)**
```
키워드: 좋, 훌륭, 완벽, 최고, 효과적, 성공
예시: "좋은 의견입니다!"
```

**negative (부정)**
```
키워드: 문제, 부족, 어렵, 위험, 취약, 결함
예시: "패스워드 정책이 너무 약한 것 같아요"
```

**neutral (중립)**
```
긍정/부정 키워드가 없거나 동일한 경우
```

### 키워드 추출

**처리 과정:**
1. 토큰화 (공백/구두점 기준)
2. 불용어 제거 (조사, 짧은 단어)
3. 어간 추출 (간단한 규칙 기반)
4. 빈도 기반 상위 키워드 선정

```
입력: "사용자 인증 부분부터 얘기를 해볼까요? 현재 패스워드 정책이 너무 약한 것 같아요"
키워드: ['사용자', '인증', '부분', '패스워드', '정책']
```

## 📊 출력 형식

### Utterance 객체

```json
{
  "utterance_id": "UTT_0001",
  "speaker": "김팀장",
  "timestamp": "00:00:05",
  "text": "네, 보안 요구사항에 대해서 논의를 시작하도록 하겠습니다.",
  "original_text": "음... 네, 그러면 오늘 오늘 보안 요구사항에 대해서...",
  "cleaned": true,
  "merged_from": []
}
```

### Intent 객체

```json
{
  "utterance_id": "UTT_0001",
  "intent_id": "INT_0001",
  "intent_type": "requirement",
  "confidence": 0.85,
  "sub_intent": "mandatory",
  "sentiment": "neutral",
  "keywords": ["보안", "요구사항", "논의", "시작"]
}
```

## 📈 성능 및 통계

### 테스트 결과 (23개 발화)

**전처리:**
- 파싱 성공률: 100%
- 잡음 제거: 필러 15개, 반복 8개
- 평균 발화 길이: 55.6자

**NLP 분석:**
- 의도 분류 정확도: 높음
- 의도 분포:
  - requirement: 30.4%
  - agreement: 30.4%
  - question: 26.1%
  - 기타: 13.1%

**감정 분석:**
- neutral: 78.3%
- positive: 21.7%
- negative: 0%

## 🛠️ 커스터마이징

### 1. 필러 단어 추가

```python
# src/preprocessing/advanced_preprocessor.py

self.filler_words = [
    '음', '어', '아', '그', '뭐',
    '커스텀필러'  # 추가
]
```

### 2. 의도 패턴 추가

```python
# src/nlp/advanced_intent_classifier.py

self.intent_patterns['new_intent'] = {
    'patterns': [
        r'새로운\s*패턴',
        r'또\s*다른\s*패턴'
    ],
    'keywords': ['키워드1', '키워드2']
}
```

### 3. 감정 키워드 추가

```python
self.sentiment_patterns['positive'].extend([
    '추가긍정', '매우좋음'
])
```

## 🔍 디버깅 및 검증

### 전처리 결과 확인

```python
# 개별 발화 확인
for utt in utterances[:5]:
    print(f"원본: {utt.original_text}")
    print(f"정제: {utt.text}")
    print()
```

### NLP 결과 확인

```python
# 의도 분석 결과 확인
for intent in intents[:5]:
    print(f"{intent.intent_type} ({intent.confidence:.2f})")
    print(f"  감정: {intent.sentiment}")
    print(f"  키워드: {intent.keywords}")
```

### 통계 확인

```python
# 화자별 통계
speaker_stats = preprocessor.get_speaker_statistics(utterances)
print(speaker_stats)

# 의도 분포
distribution = classifier.get_intent_distribution(intents)
print(distribution)
```

## 💡 활용 팁

### 1. 병합 기능 사용 시점

**병합 비활성화 (권장):**
- 발화별 세밀한 분석이 필요한 경우
- 화자 전환이 중요한 경우
- 대화 흐름 분석이 필요한 경우

**병합 활성화:**
- 요약이 목적인 경우
- 긴 문맥이 필요한 경우
- 파편화가 심한 경우

### 2. 신뢰도 활용

```python
# 높은 신뢰도 발화만 추출
high_confidence = [
    intent for intent in intents 
    if intent.confidence > 0.7
]
```

### 3. 키워드 기반 필터링

```python
# 특정 키워드 포함 발화 찾기
security_related = [
    intent for intent in intents
    if any(kw in ['보안', '인증', '암호화'] 
           for kw in intent.keywords)
]
```

## 📚 참고 자료

- **전처리 모듈**: `src/preprocessing/advanced_preprocessor.py`
- **NLP 모듈**: `src/nlp/advanced_intent_classifier.py`
- **테스트 스크립트**: `test_advanced_modules.py`
- **샘플 데이터**: `data/input/natural_meeting_transcript.txt`

## 🐛 알려진 제한사항

1. **형태소 분석**: 간단한 규칙 기반 (완전한 형태소 분석기 아님)
2. **의도 분류**: 패턴 매칭 기반 (머신러닝 모델 아님)
3. **병합 로직**: 종결 어미 기반 (완벽하지 않을 수 있음)

향후 개선 시 spaCy 또는 KoNLPy 같은 전문 라이브러리 활용 권장

---

**Version**: 1.0.0  
**Last Updated**: 2024-02-10
