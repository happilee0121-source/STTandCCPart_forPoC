"""
Layer 3. Semantic NLP Layer

주요 기능:
1. 한국어 형태소 분석 (spaCy/NLTK 대체)
2. 의도 분류 (다층 분류 모델)
3. 감정 분석
4. 핵심 키워드 추출
5. 발화 간 관계 분석
"""

import json
import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
import numpy as np
from collections import Counter, defaultdict


@dataclass
class Intent:
    """의도 분석 결과 구조"""
    utterance_id: str
    intent_id: str
    intent_type: str
    confidence: float
    sub_intent: Optional[str] = None  # 세부 의도
    sentiment: Optional[str] = None  # 감정 (positive/neutral/negative)
    keywords: List[str] = None  # 핵심 키워드
    
    def __post_init__(self):
        if self.keywords is None:
            self.keywords = []
    
    def to_dict(self) -> Dict:
        return asdict(self)


class KoreanMorphAnalyzer:
    """한국어 형태소 분석기 (간이 버전)"""
    
    def __init__(self):
        # 조사 목록
        self.josa = [
            '이', '가', '을', '를', '은', '는', '의', '에', '에서', '으로', '로',
            '과', '와', '도', '만', '부터', '까지', '께서', '에게', '한테'
        ]
        
        # 어미 패턴
        self.eomi_patterns = [
            r'습니다$', r'ㅂ니다$', r'해요$', r'요$', r'다$', r'ㅂ니까$',
            r'까요$', r'죠$', r'네요$', r'군요$', r'어요$', r'아요$'
        ]
    
    def extract_stem(self, word: str) -> str:
        """어간 추출 (간단한 규칙 기반)"""
        # 조사 제거
        for j in sorted(self.josa, key=len, reverse=True):
            if word.endswith(j) and len(word) > len(j):
                word = word[:-len(j)]
                break
        
        # 어미 제거
        for pattern in self.eomi_patterns:
            word = re.sub(pattern, '', word)
        
        return word
    
    def tokenize(self, text: str) -> List[str]:
        """간단한 토큰화"""
        # 공백 기준 분리
        tokens = text.split()
        
        # 구두점 분리
        result = []
        for token in tokens:
            # 구두점이 붙어있으면 분리
            token = re.sub(r'([.,!?])', r' \1', token)
            result.extend(token.split())
        
        return [t for t in result if t.strip()]


class AdvancedIntentClassifier:
    """고급 의도 분류 클래스"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.intent_types = config['nlp']['intent_types']
        self.morph_analyzer = KoreanMorphAnalyzer()
        
        # 의도 분류를 위한 확장된 패턴
        self.intent_patterns = {
            'requirement': {
                'patterns': [
                    r'해야\s*(합니다|해요|한다|할까요|하면)',
                    r'필요합니다|필요해요|필요하다',
                    r'적용|설정|도입|사용|구현',
                    r'반드시|꼭',
                    r'최소.*이상',
                    r'.*하도록\s*(해야|하면)',
                    r'.*게\s*해야',
                    r'권장합니다|권장해요',
                ],
                'keywords': ['적용', '설정', '도입', '필요', '해야', '반드시', '권장', '사용', '구현']
            },
            'question': {
                'patterns': [
                    r'어떻게|어떤|무엇|뭐',
                    r'\?$',
                    r'어떨까요|될까요|할까요',
                    r'가능한가요|가능할까요',
                    r'어떻게\s*생각',
                ],
                'keywords': ['어떻게', '어떤', '무엇', '뭐', '가능']
            },
            'agreement': {
                'patterns': [
                    r'^네[,.\s]|^네$',
                    r'좋습니다|좋아요|좋네요',
                    r'맞습니다|맞아요',
                    r'동의합니다|동의해요',
                    r'찬성합니다|찬성해요',
                    r'그렇습니다|그래요',
                ],
                'keywords': ['네', '좋', '맞', '동의', '찬성', '그래']
            },
            'disagreement': {
                'patterns': [
                    r'아니|아니요',
                    r'반대합니다|반대해요',
                    r'문제가|이슈가',
                    r'우려|걱정',
                    r'어렵습니다|어려워요',
                ],
                'keywords': ['아니', '반대', '문제', '우려', '걱정', '어려']
            },
            'clarification': {
                'patterns': [
                    r'즉|다시\s*말하면',
                    r'정리하면|요약하면',
                    r'예를\s*들면|예를\s*들어',
                    r'구체적으로|자세히',
                ],
                'keywords': ['즉', '정리', '요약', '예를', '구체적', '자세히']
            },
            'suggestion': {
                'patterns': [
                    r'제안|제안합니다|제안해요',
                    r'어떨까요|하는\s*게\s*어때요',
                    r'생각합니다|생각해요',
                    r'.*하면\s*좋을\s*것\s*같',
                ],
                'keywords': ['제안', '생각', '좋을', '어떨까']
            },
            'objection': {
                'patterns': [
                    r'하지만|그러나|그런데',
                    r'문제는',
                    r'고려해야|검토해야',
                ],
                'keywords': ['하지만', '그러나', '그런데', '문제', '고려', '검토']
            }
        }
        
        # 감정 분석 패턴
        self.sentiment_patterns = {
            'positive': [
                '좋', '훌륭', '완벽', '최고', '탁월', '우수', '효과적', '긍정',
                '만족', '성공', '향상', '개선'
            ],
            'negative': [
                '나쁘', '문제', '부족', '어렵', '실패', '우려', '걱정', '위험',
                '취약', '결함', '불만', '미흡'
            ]
        }
    
    def extract_keywords(self, text: str, top_k: int = 5) -> List[str]:
        """핵심 키워드 추출"""
        # 토큰화
        tokens = self.morph_analyzer.tokenize(text)
        
        # 불용어 제거 (조사, 어미, 짧은 단어)
        stopwords = ['그', '이', '저', '것', '수', '등', '및', '또한', '하는', '있는']
        
        filtered = []
        for token in tokens:
            # 구두점 제거
            if token in '.,!?':
                continue
            # 불용어 제거
            if token in stopwords:
                continue
            # 2글자 이상만
            if len(token) >= 2:
                # 어간 추출
                stem = self.morph_analyzer.extract_stem(token)
                if len(stem) >= 2:
                    filtered.append(stem)
        
        # 빈도 기반 상위 키워드 추출
        if not filtered:
            return []
        
        counter = Counter(filtered)
        keywords = [word for word, count in counter.most_common(top_k)]
        
        return keywords
    
    def classify_intent(self, text: str) -> Tuple[str, float, Optional[str]]:
        """텍스트의 의도 분류 (의도, 신뢰도, 세부의도)"""
        scores = defaultdict(float)
        
        # 각 의도 타입별 점수 계산
        for intent_type, config in self.intent_patterns.items():
            patterns = config['patterns']
            keywords = config.get('keywords', [])
            
            # 패턴 매칭 점수
            pattern_score = 0
            for pattern in patterns:
                if re.search(pattern, text):
                    pattern_score += 1.0
            
            # 키워드 매칭 점수
            keyword_score = 0
            for keyword in keywords:
                if keyword in text:
                    keyword_score += 0.5
            
            # 가중치 부여
            total_score = pattern_score * 0.7 + keyword_score * 0.3
            
            # 정규화
            if patterns:
                scores[intent_type] = min(total_score / len(patterns), 1.0)
        
        # 가장 높은 점수의 의도 선택
        if scores:
            best_intent = max(scores.items(), key=lambda x: x[1])
            intent_type, confidence = best_intent
            
            # 세부 의도 결정
            sub_intent = self._determine_sub_intent(text, intent_type)
            
            return intent_type, confidence, sub_intent
        else:
            return 'requirement', 0.5, None  # 기본값
    
    def _determine_sub_intent(self, text: str, intent_type: str) -> Optional[str]:
        """세부 의도 결정"""
        if intent_type == 'requirement':
            if '필수' in text or '반드시' in text:
                return 'mandatory'
            elif '권장' in text or '추천' in text:
                return 'recommended'
            else:
                return 'general'
        
        elif intent_type == 'question':
            if '가능' in text:
                return 'feasibility'
            elif '방법' in text or '어떻게' in text:
                return 'method'
            else:
                return 'general'
        
        return None
    
    def analyze_sentiment(self, text: str) -> str:
        """감정 분석"""
        positive_count = sum(1 for word in self.sentiment_patterns['positive'] if word in text)
        negative_count = sum(1 for word in self.sentiment_patterns['negative'] if word in text)
        
        if positive_count > negative_count:
            return 'positive'
        elif negative_count > positive_count:
            return 'negative'
        else:
            return 'neutral'
    
    def analyze_utterances(self, utterances: List[Dict]) -> List[Intent]:
        """발화 리스트에 대한 종합 분석"""
        intents = []
        
        for idx, utt in enumerate(utterances):
            text = utt['text']
            
            # 의도 분류
            intent_type, confidence, sub_intent = self.classify_intent(text)
            
            # 감정 분석
            sentiment = self.analyze_sentiment(text)
            
            # 키워드 추출
            keywords = self.extract_keywords(text, top_k=5)
            
            # Intent 객체 생성
            intent = Intent(
                utterance_id=utt['utterance_id'],
                intent_id=f"INT_{idx+1:04d}",
                intent_type=intent_type,
                confidence=confidence,
                sub_intent=sub_intent,
                sentiment=sentiment,
                keywords=keywords
            )
            
            intents.append(intent)
        
        return intents
    
    def get_intent_distribution(self, intents: List[Intent]) -> Dict:
        """의도 분포 통계"""
        distribution = defaultdict(int)
        sub_intent_dist = defaultdict(int)
        sentiment_dist = defaultdict(int)
        
        for intent in intents:
            distribution[intent.intent_type] += 1
            if intent.sub_intent:
                sub_intent_dist[f"{intent.intent_type}:{intent.sub_intent}"] += 1
            if intent.sentiment:
                sentiment_dist[intent.sentiment] += 1
        
        return {
            'intent_types': dict(distribution),
            'sub_intents': dict(sub_intent_dist),
            'sentiments': dict(sentiment_dist)
        }
    
    def extract_all_keywords(self, intents: List[Intent]) -> List[Tuple[str, int]]:
        """전체 키워드 빈도 분석"""
        all_keywords = []
        for intent in intents:
            all_keywords.extend(intent.keywords)
        
        counter = Counter(all_keywords)
        return counter.most_common(20)
    
    def save_intents(self, intents: List[Intent], output_path: str):
        """의도 분석 결과 저장"""
        distribution = self.get_intent_distribution(intents)
        top_keywords = self.extract_all_keywords(intents)
        
        data = {
            "metadata": {
                "total_intents": len(intents),
                "distribution": distribution,
                "top_keywords": [{"keyword": kw, "count": cnt} for kw, cnt in top_keywords]
            },
            "intents": [i.to_dict() for i in intents]
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def process(self, utterances_path: str, output_path: str) -> List[Intent]:
        """NLP 파이프라인 실행"""
        print(f"[NLP] 발화 데이터 로드: {utterances_path}")
        with open(utterances_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        utterances = data['utterances']
        
        print(f"[NLP] {len(utterances)}개 발화에 대한 종합 분석 중...")
        print("  - 의도 분류")
        print("  - 감정 분석")
        print("  - 키워드 추출")
        
        intents = self.analyze_utterances(utterances)
        
        print(f"[NLP] 결과 저장: {output_path}")
        self.save_intents(intents, output_path)
        
        # 통계 출력
        distribution = self.get_intent_distribution(intents)
        
        print(f"\n[NLP] 의도 분류 결과:")
        for intent_type, count in distribution['intent_types'].items():
            print(f"  - {intent_type}: {count}개")
        
        print(f"\n[NLP] 감정 분포:")
        for sentiment, count in distribution['sentiments'].items():
            print(f"  - {sentiment}: {count}개")
        
        print(f"\n[NLP] 상위 키워드:")
        top_keywords = self.extract_all_keywords(intents)[:10]
        for keyword, count in top_keywords:
            print(f"  - {keyword}: {count}회")
        
        return intents


if __name__ == "__main__":
    import yaml
    
    # 설정 로드
    with open('config/config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # NLP 실행
    classifier = AdvancedIntentClassifier(config)
    intents = classifier.process(
        utterances_path='data/output/utterances.json',
        output_path='data/output/intents.json'
    )
    
    print(f"\n[완료] 총 {len(intents)}개 의도 분석 완료")
    
    # 샘플 출력
    print(f"\n샘플 분석 결과:")
    for intent in intents[:3]:
        print(f"  - {intent.intent_type} (신뢰도: {intent.confidence:.2f})")
        print(f"    세부: {intent.sub_intent}, 감정: {intent.sentiment}")
        print(f"    키워드: {', '.join(intent.keywords[:3])}")
