"""
Semantic NLP Layer (Layer 3)
-------------------------------
목적: 전처리된 발화 텍스트를 의미 분석이 가능한 구조화 데이터로 변환
리소스: Kiwi (형태소 분석), BERT (의도 분류), KeyBERT (개념 추출)
아웃풋: Intent Model – utterance_id, intent_id, intent_type, confidence
"""

import json
import uuid
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional


# ─────────────────────────────────────────────
# 의존성 임포트 (설치 필요)
# pip install kiwipiepy keybert transformers torch sentence-transformers
# ─────────────────────────────────────────────

try:
    from kiwipiepy import Kiwi
    KIWI_AVAILABLE = True
except ImportError:
    KIWI_AVAILABLE = False
    print("[경고] kiwipiepy 미설치. 형태소 분석 스킵. → pip install kiwipiepy")

try:
    from keybert import KeyBERT
    from sentence_transformers import SentenceTransformer
    KEYBERT_AVAILABLE = True
except ImportError:
    KEYBERT_AVAILABLE = False
    print("[경고] keybert 미설치. 키워드 추출 스킵. → pip install keybert sentence-transformers")

try:
    from transformers import pipeline as hf_pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("[경고] transformers 미설치. BERT 의도분류 스킵. → pip install transformers torch")


# ─────────────────────────────────────────────
# 데이터 모델
# ─────────────────────────────────────────────

@dataclass
class MorphemeToken:
    surface: str        # 원형 표면 형태
    lemma: str          # 원형(lemma)
    pos: str            # 품사 태그 (NNG, VV, JX 등)


@dataclass
class IntentResult:
    utterance_id: str
    intent_id: str
    intent_type: str    # 보안 의도 유형
    confidence: float   # 0.0 ~ 1.0
    keywords: list[str] # KeyBERT 추출 핵심 개념어


# ─────────────────────────────────────────────
# 보안 의도 유형 정의
# ─────────────────────────────────────────────

SECURITY_INTENT_LABELS = [
    "인증_요구사항",       # 사용자 인증, 비밀번호 정책, MFA
    "접근제어_요구사항",   # 권한, 역할, 정책
    "암호화_요구사항",     # 데이터 암호화, TLS, AES
    "감사_로깅_요구사항",  # 로그, 감사 추적
    "세션_관리_요구사항",  # 세션 만료, 토큰 관리
    "네트워크_보안_요구사항", # 방화벽, API 보안
    "기타",               # 분류 불가
]


# ─────────────────────────────────────────────
# 형태소 분석기 (Kiwi)
# ─────────────────────────────────────────────

class MorphemeAnalyzer:
    """
    Kiwi 한국어 형태소 분석기 래퍼
    보안 도메인 사용자 사전 추가 지원
    """

    SECURITY_TERMS = [
        ("인증", "NNG", 0), ("암호화", "NNG", 0), ("비밀번호", "NNG", 0),
        ("세션", "NNG", 0), ("접근제어", "NNG", 0), ("방화벽", "NNG", 0),
        ("권한", "NNG", 0), ("로그", "NNG", 0), ("TLS", "SL", 0),
        ("AES", "SL", 0), ("API", "SL", 0), ("MFA", "SL", 0),
        ("감사로그", "NNG", 0), ("취약점", "NNG", 0), ("침입탐지", "NNG", 0),
    ]

    def __init__(self):
        if not KIWI_AVAILABLE:
            self.kiwi = None
            return
        self.kiwi = Kiwi()
        for term, pos, score in self.SECURITY_TERMS:
            try:
                self.kiwi.add_user_word(term, pos, score)
            except Exception:
                pass

    def analyze(self, text: str) -> list[MorphemeToken]:
        if self.kiwi is None:
            # 폴백: 공백 기준 토크나이징
            return [MorphemeToken(surface=w, lemma=w, pos="UNK") for w in text.split()]

        result = self.kiwi.tokenize(text)
        tokens = []
        for token in result:
            tokens.append(MorphemeToken(
                surface=token.form,
                lemma=token.lemma if hasattr(token, 'lemma') else token.form,
                pos=token.tag,
            ))
        return tokens

    def extract_content_words(self, tokens: list[MorphemeToken]) -> list[str]:
        """명사(N*), 동사(V*), 형용사(VA) 계열만 추출"""
        CONTENT_POS = {'NNG', 'NNP', 'NNB', 'VV', 'VA', 'XR', 'SL'}
        return [t.lemma for t in tokens if t.pos in CONTENT_POS]


# ─────────────────────────────────────────────
# 의도 분류기 (BERT)
# ─────────────────────────────────────────────

class IntentClassifier:
    """
    BERT 기반 zero-shot 의도 분류.
    한국어 지원 모델: snunlp/KR-FinBert-SC 또는 klue/roberta-base (zero-shot 가능)
    운영환경에서는 fine-tuned 모델로 교체 권장.
    """

    # Zero-shot 분류에 사용할 한국어 BERT 모델
    MODEL_NAME = "snunlp/KR-FinBert-SC"
    ZERO_SHOT_MODEL = "facebook/bart-large-mnli"  # 영어 모델 (한국어 폴백)

    def __init__(self, use_korean_model: bool = True):
        self.classifier = None
        if not TRANSFORMERS_AVAILABLE:
            return

        try:
            # 한국어 zero-shot 지원 모델 시도
            model_name = "snunlp/KR-FinBert-SC" if use_korean_model else self.ZERO_SHOT_MODEL
            self.classifier = hf_pipeline(
                "zero-shot-classification",
                model=model_name,
                device=-1,  # CPU. GPU 사용 시 device=0
            )
            print(f"[IntentClassifier] 모델 로드 완료: {model_name}")
        except Exception as e:
            print(f"[IntentClassifier] 모델 로드 실패: {e}")
            print("  → 규칙 기반 분류로 폴백합니다.")
            self.classifier = None

    def classify(self, text: str, candidate_labels: list[str]) -> tuple[str, float]:
        """
        Returns: (best_label, confidence_score)
        """
        if self.classifier:
            result = self.classifier(text, candidate_labels)
            return result['labels'][0], round(result['scores'][0], 4)
        else:
            # 규칙 기반 폴백 분류
            return self._rule_based_classify(text)

    @staticmethod
    def _rule_based_classify(text: str) -> tuple[str, float]:
        """키워드 매칭 기반 규칙 분류 (BERT 사용 불가 시 폴백)"""
        rules = {
            "인증_요구사항": ["인증", "로그인", "비밀번호", "패스워드", "MFA", "2단계", "계정"],
            "접근제어_요구사항": ["접근", "권한", "역할", "관리자", "분리", "허가", "차단"],
            "암호화_요구사항": ["암호화", "TLS", "AES", "SSL", "복호화", "해시", "키"],
            "감사_로깅_요구사항": ["로그", "감사", "기록", "추적", "이력", "모니터링"],
            "세션_관리_요구사항": ["세션", "만료", "토큰", "쿠키", "타임아웃"],
            "네트워크_보안_요구사항": ["API", "방화벽", "네트워크", "포트", "프로토콜", "HTTPS"],
        }
        scores: dict[str, int] = {}
        for intent, keywords in rules.items():
            scores[intent] = sum(1 for kw in keywords if kw in text)

        best_intent = max(scores, key=lambda k: scores[k])
        best_score = scores[best_intent]

        if best_score == 0:
            return "기타", 0.5

        # 총 점수 대비 비율로 신뢰도 계산
        total = sum(scores.values())
        confidence = round(best_score / total, 4) if total > 0 else 0.5
        return best_intent, confidence


# ─────────────────────────────────────────────
# 키워드 추출기 (KeyBERT)
# ─────────────────────────────────────────────

class SecurityKeywordExtractor:
    """
    KeyBERT를 활용한 보안 핵심 개념어 추출.
    한국어 임베딩: snunlp/KR-ELECTRA-discriminator 또는 paraphrase-multilingual-MiniLM-L12-v2
    """

    MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

    def __init__(self):
        self.kw_model = None
        if not KEYBERT_AVAILABLE:
            return
        try:
            self.kw_model = KeyBERT(model=self.MODEL_NAME)
            print(f"[KeywordExtractor] 모델 로드 완료: {self.MODEL_NAME}")
        except Exception as e:
            print(f"[KeywordExtractor] 모델 로드 실패: {e}")

    def extract(self, text: str, top_n: int = 5) -> list[str]:
        if self.kw_model is None:
            # 폴백: 명사 추출 (단순 규칙)
            return self._fallback_extract(text, top_n)

        keywords = self.kw_model.extract_keywords(
            text,
            keyphrase_ngram_range=(1, 2),
            stop_words=None,
            top_n=top_n,
            use_mmr=True,       # 다양성 확보
            diversity=0.5,
        )
        return [kw for kw, _ in keywords]

    @staticmethod
    def _fallback_extract(text: str, top_n: int) -> list[str]:
        """형태소 분석 없이 보안 키워드 매칭으로 추출"""
        security_vocab = [
            "인증", "암호화", "접근제어", "세션", "권한", "방화벽",
            "TLS", "AES", "API", "로그", "감사", "비밀번호", "MFA",
            "취약점", "침입", "데이터", "네트워크", "프로토콜",
        ]
        found = [kw for kw in security_vocab if kw in text]
        return found[:top_n]


# ─────────────────────────────────────────────
# NLP 파이프라인 통합
# ─────────────────────────────────────────────

class SemanticNLPPipeline:
    """
    Semantic NLP 파이프라인:
    1. 형태소 분석 (Kiwi)
    2. 의도 분류 (BERT zero-shot)
    3. 핵심 개념 추출 (KeyBERT)
    4. IntentResult 구조화
    """

    def __init__(self):
        print("[NLP Pipeline] 모델 초기화 중...")
        self.morpheme_analyzer = MorphemeAnalyzer()
        self.intent_classifier = IntentClassifier()
        self.keyword_extractor = SecurityKeywordExtractor()
        print("[NLP Pipeline] 초기화 완료\n")

    def process_utterance(self, utterance: dict) -> IntentResult:
        uid = utterance["utterance_id"]
        text = utterance["text"]

        # Step 1: 형태소 분석
        tokens = self.morpheme_analyzer.analyze(text)
        content_words = self.morpheme_analyzer.extract_content_words(tokens)
        enriched_text = text  # 원문 기반으로 분류 (형태소 결과는 보조)

        # Step 2: 의도 분류
        intent_type, confidence = self.intent_classifier.classify(
            enriched_text, SECURITY_INTENT_LABELS
        )

        # Step 3: 키워드 추출
        keywords = self.keyword_extractor.extract(text, top_n=5)

        return IntentResult(
            utterance_id=uid,
            intent_id=str(uuid.uuid4()),
            intent_type=intent_type,
            confidence=confidence,
            keywords=keywords,
        )

    def run(self, utterances: list[dict]) -> list[IntentResult]:
        results: list[IntentResult] = []
        for i, u in enumerate(utterances):
            print(f"[NLP] 처리 중 ({i+1}/{len(utterances)}): {u['text'][:40]}...")
            result = self.process_utterance(u)
            results.append(result)
        print(f"\n[NLP] {len(results)}개 발화 의도 분류 완료")
        return results

    def run_from_json(self, json_path: str) -> list[IntentResult]:
        """전처리 레이어 아웃풋(JSON) 파일에서 직접 로드"""
        data = json.loads(Path(json_path).read_text(encoding='utf-8'))
        return self.run(data)

    def to_json(self, results: list[IntentResult], output_path: Optional[str] = None) -> str:
        data = [asdict(r) for r in results]
        json_str = json.dumps(data, ensure_ascii=False, indent=2)
        if output_path:
            Path(output_path).write_text(json_str, encoding='utf-8')
            print(f"[Output] Intent Model JSON 저장 완료: {output_path}")
        return json_str


# ─────────────────────────────────────────────
# 실행 예시
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    # 전처리 결과 JSON 로드 (직접 실행 시 샘플 사용)
    SAMPLE_UTTERANCES = [
        {
            "utterance_id": "aaa-001",
            "speaker": "화자1",
            "timestamp": "00:00:22",
            "timestamp_sec": 22.0,
            "text": "로그인 시 비밀번호는 반드시 암호화해서 저장해야 합니다. 그리고 2단계 인증도 지원해야 합니다."
        },
        {
            "utterance_id": "aaa-002",
            "speaker": "화자3",
            "timestamp": "00:00:45",
            "timestamp_sec": 45.0,
            "text": "세션 만료 시간을 설정해야 할 것 같습니다."
        },
        {
            "utterance_id": "aaa-003",
            "speaker": "화자2",
            "timestamp": "00:00:58",
            "timestamp_sec": 58.0,
            "text": "관리자 계정은 별도 권한 분리가 필요합니다."
        },
        {
            "utterance_id": "aaa-004",
            "speaker": "화자3",
            "timestamp": "00:01:12",
            "timestamp_sec": 72.0,
            "text": "전송 중인 데이터는 TLS를 사용해야 하고, 저장 데이터는 AES-256 이상을 적용해야 합니다."
        },
        {
            "utterance_id": "aaa-005",
            "speaker": "화자2",
            "timestamp": "00:01:35",
            "timestamp_sec": 95.0,
            "text": "모든 접근 이력을 감사 로그로 기록해야 합니다."
        },
    ]

    # JSON 파일 인자 있으면 파일에서 로드
    input_path = sys.argv[1] if len(sys.argv) > 1 else None

    pipeline = SemanticNLPPipeline()

    if input_path:
        results = pipeline.run_from_json(input_path)
    else:
        results = pipeline.run(SAMPLE_UTTERANCES)

    print("\n=== Intent Model (미리보기) ===")
    for r in results:
        print(f"[{r.utterance_id[:8]}...] 의도: {r.intent_type} (신뢰도: {r.confidence:.2f}) | 키워드: {r.keywords}")

    # JSON 저장
    pipeline.to_json(results, "intent_model.json")
