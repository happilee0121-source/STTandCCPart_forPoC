"""
AI Harness 기반 보안 요구사항 추출 및 CC(Common Criteria) 매핑 시스템

파이프라인:
[STT Input] → [NLP Filter] → [Intent Filter] → [Requirement Extractor]
→ [Requirement Validator] → [CC Candidate Retrieval] → [Rule-based Filter]
→ [LLM Selector] → [Confidence Scoring] → [Missing Requirement Detector]
→ [Structured Output + Explanation]
"""

import json
import re
import math
import hashlib
from dataclasses import dataclass, field, asdict
from typing import Optional
from enum import Enum


# 외부 라이브러리 임포트 (없으면 mock으로 대체)

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    print("[WARNING] anthropic 패키지 없음 → LLM 호출 mock 모드로 실행")

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    print("[WARNING] numpy 없음 → 벡터 유사도 수식 구현으로 대체")

try:
    import jsonschema
    JSONSCHEMA_AVAILABLE = True
except ImportError:
    JSONSCHEMA_AVAILABLE = False
    print("[WARNING] jsonschema 없음 → 내장 검증 로직 사용")

try:
    import z3
    Z3_AVAILABLE = True
except ImportError:
    Z3_AVAILABLE = False
    print("[WARNING] z3-solver 없음 → 규칙 기반 의미 검증으로 대체")


# 데이터 스키마 정의

class IntentType(str, Enum):
    REQUIREMENT  = "requirement"
    OPINION      = "opinion"
    EXPLANATION  = "explanation"
    DECISION     = "decision"
    OTHER        = "other"

class RequirementType(str, Enum):
    FUNCTIONAL   = "functional"
    NON_FUNCTIONAL = "non_functional"
    SECURITY     = "security"
    COMPLIANCE   = "compliance"

@dataclass
class SecurityRequirement:
    req_id:      str
    actor:       str
    action:      str
    object:      str
    req_type:    str
    raw_text:    str
    confidence:  float = 0.0
    valid:       bool  = False
    issues:      list  = field(default_factory=list)

@dataclass
class CCMapping:
    req_id:       str
    cc_component: str
    cc_family:    str
    cc_class:     str
    description:  str
    confidence:   float
    rationale:    str

@dataclass
class PipelineResult:
    requirements:        list[SecurityRequirement]
    cc_mappings:         list[CCMapping]
    missing_components:  list[dict]
    coverage_score:      float
    mapping_matrix:      dict
    explanation:         dict = field(default_factory=dict)


# CC Part2 SFR 지식베이스 (내장 RAG 데이터베이스)

CC_SFR_DATABASE = [
    # FAU – Security Audit
    {"component": "FAU_GEN.1", "family": "FAU_GEN", "cc_class": "FAU",
     "description": "Audit data generation: The TSF shall be able to generate an audit record of auditable events.",
     "keywords": ["audit", "log", "record", "event", "generate", "logging", "감사", "로그", "기록"]},
    {"component": "FAU_GEN.2", "family": "FAU_GEN", "cc_class": "FAU",
     "description": "User identity association: The TSF shall be able to associate each auditable event with the identity of the user.",
     "keywords": ["user identity", "audit", "association", "사용자 식별", "감사"]},
    {"component": "FAU_SAR.1", "family": "FAU_SAR", "cc_class": "FAU",
     "description": "Audit review: The TSF shall provide authorised users with the capability to read audit information.",
     "keywords": ["audit review", "read audit", "authorised", "감사 검토", "감사 읽기"]},
    {"component": "FAU_STG.1", "family": "FAU_STG", "cc_class": "FAU",
     "description": "Protected audit trail storage: The TSF shall protect the stored audit records from unauthorised deletion.",
     "keywords": ["audit storage", "protect", "deletion", "감사 저장", "보호"]},

    # FIA – Identification and Authentication
    {"component": "FIA_UAU.1", "family": "FIA_UAU", "cc_class": "FIA",
     "description": "Timing of authentication: The TSF shall allow user actions before authentication.",
     "keywords": ["authentication", "timing", "before", "인증", "시기"]},
    {"component": "FIA_UAU.2", "family": "FIA_UAU", "cc_class": "FIA",
     "description": "User authentication before any action: The TSF shall require each user to be successfully authenticated before any TSF-mediated actions.",
     "keywords": ["authentication", "user", "action", "사용자 인증", "접근 전 인증", "login", "로그인"]},
    {"component": "FIA_UAU.5", "family": "FIA_UAU", "cc_class": "FIA",
     "description": "Multiple authentication mechanisms: The TSF shall provide multiple authentication mechanisms to support user authentication.",
     "keywords": ["multi-factor", "MFA", "multiple authentication", "다중 인증", "이중 인증"]},
    {"component": "FIA_UID.1", "family": "FIA_UID", "cc_class": "FIA",
     "description": "Timing of identification: The TSF shall allow user identification before authentication.",
     "keywords": ["identification", "user id", "identity", "식별", "아이디"]},
    {"component": "FIA_UID.2", "family": "FIA_UID", "cc_class": "FIA",
     "description": "User identification before any action: The TSF shall require each user to identify themselves before any actions.",
     "keywords": ["identification", "before action", "사용자 식별", "행위 전 식별"]},
    {"component": "FIA_ATD.1", "family": "FIA_ATD", "cc_class": "FIA",
     "description": "User attribute definition: The TSF shall maintain a list of security attributes belonging to individual users.",
     "keywords": ["attribute", "user attribute", "security attribute", "사용자 속성"]},

    # FDP – User Data Protection
    {"component": "FDP_ACC.1", "family": "FDP_ACC", "cc_class": "FDP",
     "description": "Subset access control: The TSF shall enforce the access control SFP on subjects, objects and operations.",
     "keywords": ["access control", "subject", "object", "접근 제어", "권한"]},
    {"component": "FDP_ACC.2", "family": "FDP_ACC", "cc_class": "FDP",
     "description": "Complete access control: The TSF shall enforce the access control SFP on all operations.",
     "keywords": ["complete access control", "all operations", "완전한 접근 제어"]},
    {"component": "FDP_ACF.1", "family": "FDP_ACF", "cc_class": "FDP",
     "description": "Security attribute based access control: The TSF shall enforce the access control SFP based on security attributes.",
     "keywords": ["attribute based", "access control", "security attributes", "속성 기반 접근 제어"]},
    {"component": "FDP_IFF.1", "family": "FDP_IFF", "cc_class": "FDP",
     "description": "Simple security attributes: The TSF shall enforce the information flow control SFP.",
     "keywords": ["information flow", "data flow", "정보 흐름", "데이터 흐름 제어"]},
    {"component": "FDP_ETC.1", "family": "FDP_ETC", "cc_class": "FDP",
     "description": "Export of user data without security attributes: The TSF shall enforce the SFP when exporting user data.",
     "keywords": ["export", "data export", "데이터 내보내기", "반출"]},
    {"component": "FDP_UCT.1", "family": "FDP_UCT", "cc_class": "FDP",
     "description": "Basic data exchange confidentiality: The TSF shall enforce the access control SFP to be able to transmit objects in a manner protected from unauthorised disclosure.",
     "keywords": ["confidentiality", "data exchange", "encrypt", "암호화", "기밀성", "전송 보호"]},

    # FCO – Communication
    {"component": "FCO_NRO.1", "family": "FCO_NRO", "cc_class": "FCO",
     "description": "Selective proof of origin: The TSF shall be able to generate evidence of the origin of transmitted information.",
     "keywords": ["non-repudiation", "origin", "proof", "부인방지", "출처 증명"]},
    {"component": "FCO_NRR.1", "family": "FCO_NRR", "cc_class": "FCO",
     "description": "Selective proof of receipt: The TSF shall be able to generate evidence of the receipt of information.",
     "keywords": ["non-repudiation", "receipt", "부인방지", "수신 증명"]},

    # FCS – Cryptographic Support
    {"component": "FCS_CKM.1", "family": "FCS_CKM", "cc_class": "FCS",
     "description": "Cryptographic key generation: The TSF shall generate cryptographic keys in accordance with a specified algorithm.",
     "keywords": ["key generation", "cryptographic", "암호화 키", "키 생성"]},
    {"component": "FCS_CKM.4", "family": "FCS_CKM", "cc_class": "FCS",
     "description": "Cryptographic key destruction: The TSF shall destroy cryptographic keys in accordance with a specified key destruction method.",
     "keywords": ["key destruction", "key management", "키 폐기", "키 관리"]},
    {"component": "FCS_COP.1", "family": "FCS_COP", "cc_class": "FCS",
     "description": "Cryptographic operation: The TSF shall perform cryptographic operations in accordance with a specified algorithm.",
     "keywords": ["encryption", "decryption", "cryptographic operation", "암호 연산", "복호화", "암호화"]},

    # FMT – Security Management
    {"component": "FMT_MSA.1", "family": "FMT_MSA", "cc_class": "FMT",
     "description": "Management of security attributes: The TSF shall enforce the access control SFP to restrict the ability to manage security attributes.",
     "keywords": ["security attributes management", "보안 속성 관리"]},
    {"component": "FMT_MSA.3", "family": "FMT_MSA", "cc_class": "FMT",
     "description": "Static attribute initialisation: The TSF shall enforce the access control SFP to provide restrictive default values.",
     "keywords": ["default values", "initialisation", "기본값 설정", "초기화"]},
    {"component": "FMT_MTD.1", "family": "FMT_MTD", "cc_class": "FMT",
     "description": "Management of TSF data: The TSF shall restrict the ability to manage TSF data to authorised users.",
     "keywords": ["TSF data management", "administrator", "관리자", "TSF 데이터 관리"]},
    {"component": "FMT_SMF.1", "family": "FMT_SMF", "cc_class": "FMT",
     "description": "Specification of management functions: The TSF shall be capable of performing security management functions.",
     "keywords": ["management functions", "security management", "보안 관리 기능"]},
    {"component": "FMT_SMR.1", "family": "FMT_SMR", "cc_class": "FMT",
     "description": "Security roles: The TSF shall maintain the roles and associate users with roles.",
     "keywords": ["roles", "role management", "RBAC", "역할", "역할 기반"]},

    # FPT – Protection of TSF
    {"component": "FPT_STM.1", "family": "FPT_STM", "cc_class": "FPT",
     "description": "Reliable time stamps: The TSF shall be able to provide reliable time stamps.",
     "keywords": ["time stamp", "timestamp", "시간 도장", "타임스탬프"]},
    {"component": "FPT_TST.1", "family": "FPT_TST", "cc_class": "FPT",
     "description": "TSF testing: The TSF shall run a suite of self tests to demonstrate the correct operation.",
     "keywords": ["self test", "integrity test", "자가 검사", "무결성 테스트"]},
    {"component": "FPT_ITT.1", "family": "FPT_ITT", "cc_class": "FPT",
     "description": "Basic internal TSF data transfer protection: The TSF shall protect TSF data from disclosure during transmission.",
     "keywords": ["internal transfer", "data protection", "내부 전송 보호"]},

    # FTA – TOE Access
    {"component": "FTA_SSL.1", "family": "FTA_SSL", "cc_class": "FTA",
     "description": "TSF-initiated session locking: The TSF shall lock an interactive session after a specified time of user inactivity.",
     "keywords": ["session lock", "timeout", "inactivity", "세션 잠금", "타임아웃"]},
    {"component": "FTA_SSL.3", "family": "FTA_SSL", "cc_class": "FTA",
     "description": "TSF-initiated termination: The TSF shall terminate an interactive session after a specified time of user inactivity.",
     "keywords": ["session termination", "session timeout", "세션 종료"]},
    {"component": "FTA_TAH.1", "family": "FTA_TAH", "cc_class": "FTA",
     "description": "TOE access history: The TSF shall display history of successful and unsuccessful access attempts to the TOE.",
     "keywords": ["access history", "login history", "접근 이력", "로그인 이력"]},

    # FRU – Resource Utilization
    {"component": "FRU_FLT.1", "family": "FRU_FLT", "cc_class": "FRU",
     "description": "Degraded fault tolerance: The TSF shall ensure the operation of some security capabilities when faults occur.",
     "keywords": ["fault tolerance", "availability", "결함 허용", "가용성"]},
    {"component": "FRU_RSA.1", "family": "FRU_RSA", "cc_class": "FRU",
     "description": "Maximum quotas: The TSF shall enforce maximum quotas of resources used by subjects.",
     "keywords": ["resource quota", "maximum quota", "자원 할당", "쿼터"]},
]


# CC 도메인 규칙 (Rule-based Filter)

DOMAIN_RULES = {
    "authentication":   ["FIA_UAU", "FIA_UID", "FIA_ATD"],
    "authorization":    ["FDP_ACC", "FDP_ACF", "FMT_MSA", "FMT_SMR"],
    "audit":            ["FAU_GEN", "FAU_SAR", "FAU_STG"],
    "encryption":       ["FCS_CKM", "FCS_COP", "FDP_UCT"],
    "session":          ["FTA_SSL", "FTA_TAH"],
    "data_protection":  ["FDP_ACC", "FDP_IFF", "FDP_ETC", "FDP_UCT"],
    "non_repudiation":  ["FCO_NRO", "FCO_NRR"],
    "management":       ["FMT_MSA", "FMT_MTD", "FMT_SMF", "FMT_SMR"],
    "integrity":        ["FPT_TST", "FPT_ITT", "FPT_STM"],
    "availability":     ["FRU_FLT", "FRU_RSA"],
}

SECURITY_KEYWORD_DOMAIN_MAP = {
    "인증": "authentication",     "authentication": "authentication",
    "로그인": "authentication",   "login": "authentication",
    "비밀번호": "authentication",  "password": "authentication",
    "접근": "authorization",      "access": "authorization",
    "권한": "authorization",      "permission": "authorization",
    "역할": "authorization",      "role": "authorization",
    "감사": "audit",              "audit": "audit",
    "로그": "audit",              "log": "audit",
    "기록": "audit",              "record": "audit",
    "암호화": "encryption",       "encrypt": "encryption",
    "복호화": "encryption",       "decrypt": "encryption",
    "세션": "session",            "session": "session",
    "타임아웃": "session",        "timeout": "session",
    "데이터 보호": "data_protection",
    "부인방지": "non_repudiation", "non-repudiation": "non_repudiation",
    "무결성": "integrity",        "integrity": "integrity",
    "가용성": "availability",     "availability": "availability",
}


# JSON Schema (Requirement Extractor 강제 구조)

REQUIREMENT_JSON_SCHEMA = {
    "type": "object",
    "required": ["req_id", "actor", "action", "object", "req_type", "raw_text"],
    "properties": {
        "req_id":    {"type": "string", "pattern": "^REQ-\\d{3}$"},
        "actor":     {"type": "string", "minLength": 1},
        "action":    {"type": "string", "minLength": 1},
        "object":    {"type": "string", "minLength": 1},
        "req_type":  {"type": "string", "enum": ["functional", "non_functional", "security", "compliance"]},
        "raw_text":  {"type": "string", "minLength": 5},
        "confidence":{"type": "number", "minimum": 0.0, "maximum": 1.0},
    },
    "additionalProperties": False,
}


# 유틸리티 함수

def cosine_similarity_pure(vec_a: list[float], vec_b: list[float]) -> float:
    """numpy 없이 코사인 유사도 계산"""
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    mag_a = math.sqrt(sum(a * a for a in vec_a))
    mag_b = math.sqrt(sum(b * b for b in vec_b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)

def simple_tfidf_vector(text: str, vocab: list[str]) -> list[float]:
    """단순 단어 존재 여부 기반 벡터 (TF-IDF 근사)"""
    text_lower = text.lower()
    return [1.0 if word.lower() in text_lower else 0.0 for word in vocab]

def build_vocab(texts: list[str]) -> list[str]:
    """어휘 사전 구성"""
    vocab = set()
    for text in texts:
        tokens = re.findall(r'[a-zA-Z가-힣]+', text.lower())
        vocab.update(tokens)
    return sorted(vocab)


# STAGE 1: NLP Filter

class NLPFilter:
    """발화 의미 단위화 및 노이즈 제거"""

    STOP_WORDS = {
        "그리고", "그런데", "그러나", "하지만", "또한", "따라서", "즉",
        "음", "어", "뭐", "약", "잠깐", "어디", "저기", "글쎄",
        "and", "or", "but", "so", "well", "um", "uh", "like", "you know",
    }

    NOISE_PATTERNS = [
        r'\[.*?\]',          # STT 메타데이터 (예: [웃음], [박수])
        r'\(.*?\)',          # 발화자 표기
        r'\.{3,}',           # 말줄임
        r'\s{2,}',           # 중복 공백
    ]

    def process(self, raw_text: str) -> list[str]:
        """원시 STT 텍스트를 정제된 의미 단위로 분리"""
        print(f"\n[NLP Filter] 입력 길이: {len(raw_text)}자")

        # 1. 노이즈 패턴 제거
        cleaned = raw_text
        for pattern in self.NOISE_PATTERNS:
            cleaned = re.sub(pattern, ' ', cleaned)
        cleaned = cleaned.strip()

        # 2. 줄바꿈 기준 1차 분리 후, 문장 부호 기준 2차 분리
        lines = [line.strip() for line in cleaned.split('\n') if line.strip()]
        raw_sentences = []
        for line in lines:
            # 번호 목록 패턴 제거 (예: "1. ", "2. ")
            line = re.sub(r'^\d+\.\s*', '', line)
            # 문장 부호 기준 분리
            parts = re.split(r'(?<=[.!?。])\s+', line)
            raw_sentences.extend(parts)
        sentences = raw_sentences

        # 3. 불용어 제거 및 의미 단위 필터링
        units = []
        for sent in sentences:
            sent = sent.strip()
            if len(sent) < 5:
                continue
            # 불용어만으로 구성된 문장 제거
            tokens = re.findall(r'[a-zA-Z가-힣]+', sent)
            meaningful = [t for t in tokens if t not in self.STOP_WORDS]
            if len(meaningful) >= 2:
                units.append(sent)

        print(f"[NLP Filter] 의미 단위 {len(units)}개 추출")
        return units


# STAGE 2: Intent Filter

class IntentFilter:
    """발화 의도 분류: requirement / opinion / explanation / decision / other"""

    REQUIREMENT_SIGNALS = [
        r'해야\s*(한다|합니다|됩니다|함)',
        r'(필요|필수|의무)',
        r'(shall|must|should|required|need to)',
        r'(기능|요구|구현|적용|지원|제공).*?(해야|한다|됩니다)',
        r'(보안|인증|암호화|접근제어|감사|권한).*?(해야|한다|필요)',
    ]
    OPINION_SIGNALS = [
        r'(좋겠|좋을\s*것|좋아|생각|의견|제안|바람직)',
        r'(think|believe|suggest|prefer|opinion|feel)',
    ]
    EXPLANATION_SIGNALS = [
        r'(왜냐하면|때문에|이유는|즉|설명|의미)',
        r'(because|since|reason|means|explain)',
    ]
    DECISION_SIGNALS = [
        r'(결정|확정|합의|승인|채택|결론)',
        r'(decided|agreed|approved|confirmed|conclusion)',
    ]

    def _match(self, text: str, patterns: list[str]) -> bool:
        for p in patterns:
            if re.search(p, text, re.IGNORECASE):
                return True
        return False

    def classify(self, units: list[str]) -> list[tuple[str, IntentType]]:
        """각 의미 단위에 의도 유형 부여"""
        results = []
        req_count = 0
        for unit in units:
            if self._match(unit, self.REQUIREMENT_SIGNALS):
                intent = IntentType.REQUIREMENT
                req_count += 1
            elif self._match(unit, self.DECISION_SIGNALS):
                intent = IntentType.DECISION
            elif self._match(unit, self.EXPLANATION_SIGNALS):
                intent = IntentType.EXPLANATION
            elif self._match(unit, self.OPINION_SIGNALS):
                intent = IntentType.OPINION
            else:
                intent = IntentType.OTHER
            results.append((unit, intent))
        print(f"[Intent Filter] 요구사항 {req_count}개 / 전체 {len(units)}개")
        return results


# STAGE 3: Requirement Extractor

class RequirementExtractor:
    """자연어 → JSON Schema 강제 구조 추출"""

    ACTOR_PATTERNS = [
        (r'(시스템|system)', "시스템"),
        (r'(관리자|admin(?:istrator)?)', "관리자"),
        (r'(사용자|user)', "사용자"),
        (r'(TSF|TOE)', "TSF"),
        (r'(애플리케이션|application|앱|app)', "애플리케이션"),
        (r'(서버|server)', "서버"),
    ]
    ACTION_PATTERNS = [
        (r'(인증|authenticate|authentication)', "인증"),
        (r'(암호화|encrypt(?:ion)?)', "암호화"),
        (r'(복호화|decrypt(?:ion)?)', "복호화"),
        (r'(로그|기록|audit|log(?:ging)?)', "로그 기록"),
        (r'(접근 제어|access control)', "접근 제어"),
        (r'(검증|validate|validation|verify)', "검증"),
        (r'(관리|manage|management)', "관리"),
        (r'(제공|provide|지원|support)', "제공"),
        (r'(차단|block|deny)', "차단"),
        (r'(모니터링|monitor(?:ing)?)', "모니터링"),
        (r'(보호|protect(?:ion)?)', "보호"),
        (r'(저장|store|storage)', "저장"),
        (r'(전송|transmit|transfer)', "전송"),
        (r'(생성|generate|create)', "생성"),
        (r'(삭제|delete|destroy)', "삭제"),
    ]
    OBJECT_PATTERNS = [
        (r'(비밀번호|패스워드|password)', "비밀번호"),
        (r'(세션|session)', "세션"),
        (r'(데이터|data)', "데이터"),
        (r'(로그|감사 기록|audit log)', "감사 로그"),
        (r'(키|key|암호키)', "암호화 키"),
        (r'(접근 권한|권한|permission|privilege)', "접근 권한"),
        (r'(사용자 정보|user (data|info(?:rmation)?))', "사용자 정보"),
        (r'(통신|communication|네트워크|network)', "통신 채널"),
        (r'(이벤트|event)', "이벤트"),
        (r'(파일|file)', "파일"),
        (r'(역할|role)', "역할"),
    ]
    TYPE_KEYWORDS = {
        RequirementType.SECURITY: ["보안", "security", "인증", "암호화", "권한", "접근"],
        RequirementType.COMPLIANCE: ["컴플라이언스", "compliance", "규정", "standard", "CC", "표준"],
        RequirementType.FUNCTIONAL: ["기능", "functional", "제공", "지원", "처리"],
        RequirementType.NON_FUNCTIONAL: ["성능", "performance", "가용성", "availability", "확장성"],
    }

    def _extract_field(self, text: str, patterns: list[tuple]) -> str:
        for pattern, value in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return value
        return "unknown"

    def _infer_type(self, text: str) -> str:
        scores = {t: 0 for t in RequirementType}
        for req_type, keywords in self.TYPE_KEYWORDS.items():
            for kw in keywords:
                if kw.lower() in text.lower():
                    scores[req_type] += 1
        best = max(scores, key=lambda t: scores[t])
        return best.value if scores[best] > 0 else RequirementType.SECURITY.value

    def extract(self, classified: list[tuple[str, IntentType]]) -> list[SecurityRequirement]:
        """요구사항 의도를 가진 단위에서 구조화된 요구사항 추출"""
        requirements = []
        req_idx = 1
        for text, intent in classified:
            if intent != IntentType.REQUIREMENT:
                continue
            req = SecurityRequirement(
                req_id   = f"REQ-{req_idx:03d}",
                actor    = self._extract_field(text, self.ACTOR_PATTERNS),
                action   = self._extract_field(text, self.ACTION_PATTERNS),
                object   = self._extract_field(text, self.OBJECT_PATTERNS),
                req_type = self._infer_type(text),
                raw_text = text.strip(),
            )
            requirements.append(req)
            req_idx += 1
        print(f"[Requirement Extractor] {len(requirements)}개 요구사항 구조화")
        return requirements


# STAGE 4: Requirement Validator

class RequirementValidator:
    """JSON Schema 형식 검증 + z3 기반 의미 검증"""

    def _validate_schema(self, req: SecurityRequirement) -> list[str]:
        """JSON Schema 형식 검증"""
        issues = []
        d = asdict(req)
        # 필수 필드 검증
        for field_name in ["req_id", "actor", "action", "object", "req_type", "raw_text"]:
            if not d.get(field_name) or d[field_name] == "unknown":
                issues.append(f"필드 '{field_name}' 값 없음 또는 식별 불가")
        # req_id 패턴 검증
        if not re.match(r'^REQ-\d{3}$', req.req_id):
            issues.append(f"req_id 형식 오류: {req.req_id}")
        # req_type 열거값 검증
        valid_types = [t.value for t in RequirementType]
        if req.req_type not in valid_types:
            issues.append(f"req_type '{req.req_type}'은 유효하지 않은 값")
        # jsonschema 라이브러리 사용 가능 시 추가 검증
        if JSONSCHEMA_AVAILABLE:
            try:
                obj = {k: v for k, v in asdict(req).items() if k in REQUIREMENT_JSON_SCHEMA["properties"]}
                jsonschema.validate(instance=obj, schema=REQUIREMENT_JSON_SCHEMA)
            except jsonschema.ValidationError as e:
                issues.append(f"JSON Schema 오류: {e.message}")
        return issues

    def _validate_semantics_z3(self, req: SecurityRequirement) -> list[str]:
        """z3 solver를 이용한 의미적 타당성 검증"""
        issues = []
        if not Z3_AVAILABLE:
            # z3 없을 때 규칙 기반 대체 검증
            return self._validate_semantics_rule(req)

        try:
            # z3 Boolean 변수 정의
            has_actor  = z3.Bool("has_actor")
            has_action = z3.Bool("has_action")
            has_object = z3.Bool("has_object")
            is_valid   = z3.Bool("is_valid")

            s = z3.Solver()
            # 유효한 요구사항 조건: actor & action & object 모두 존재
            s.add(is_valid == z3.And(has_actor, has_action, has_object))
            s.add(has_actor  == (req.actor  != "unknown"))
            s.add(has_action == (req.action != "unknown"))
            s.add(has_object == (req.object != "unknown"))
            s.add(is_valid)  # 유효해야 한다는 가정

            result = s.check()
            if result == z3.unsat:
                issues.append("z3 검증 실패: actor/action/object 중 하나 이상 미식별")
        except Exception as e:
            issues.append(f"z3 오류: {e}")
        return issues

    def _validate_semantics_rule(self, req: SecurityRequirement) -> list[str]:
        """z3 없을 때 규칙 기반 의미 검증"""
        issues = []
        if req.actor == "unknown":
            issues.append("의미 검증: 주체(actor) 식별 불가 — 요구사항이 모호함")
        if req.action == "unknown":
            issues.append("의미 검증: 행위(action) 식별 불가 — 동작이 명확하지 않음")
        if req.object == "unknown":
            issues.append("의미 검증: 객체(object) 식별 불가 — 대상이 특정되지 않음")
        if len(req.raw_text.split()) < 3:
            issues.append("의미 검증: 요구사항 문장이 너무 짧아 의미 파악 어려움")
        return issues

    def validate(self, requirements: list[SecurityRequirement]) -> list[SecurityRequirement]:
        """모든 요구사항에 대해 형식 + 의미 검증 수행"""
        valid_count = 0
        for req in requirements:
            schema_issues   = self._validate_schema(req)
            semantic_issues = self._validate_semantics_z3(req)
            req.issues = schema_issues + semantic_issues
            req.valid  = len(req.issues) == 0
            if req.valid:
                valid_count += 1
        print(f"[Requirement Validator] {valid_count}/{len(requirements)}개 유효")
        return requirements


# STAGE 5: CC Candidate Retrieval (RAG)

class CCCandidateRetriever:
    """RAG 임베딩 데이터베이스 기반 CC SFR 후보 검색"""

    def __init__(self):
        # 모든 CC SFR 텍스트로부터 어휘 사전 구성
        all_texts = [
            " ".join(item["keywords"]) + " " + item["description"]
            for item in CC_SFR_DATABASE
        ]
        self.vocab = build_vocab(all_texts)
        # CC SFR 벡터 사전 구성
        self.sfr_vectors = {}
        for item in CC_SFR_DATABASE:
            text = " ".join(item["keywords"]) + " " + item["description"]
            self.sfr_vectors[item["component"]] = simple_tfidf_vector(text, self.vocab)

    def _vectorize(self, text: str) -> list[float]:
        return simple_tfidf_vector(text, self.vocab)

    def retrieve(self, req: SecurityRequirement, top_k: int = 5) -> list[dict]:
        """요구사항과 가장 유사한 CC SFR 항목 top_k개 반환"""
        query = f"{req.action} {req.object} {req.actor} {req.raw_text}"
        q_vec = self._vectorize(query)

        scores = []
        for item in CC_SFR_DATABASE:
            sfr_vec = self.sfr_vectors[item["component"]]
            sim = cosine_similarity_pure(q_vec, sfr_vec)
            # 키워드 직접 매칭 보너스
            kw_bonus = sum(0.15 for kw in item["keywords"]
                          if kw.lower() in query.lower())
            scores.append((item, min(sim + kw_bonus, 1.0)))

        scores.sort(key=lambda x: x[1], reverse=True)
        candidates = [
            {**item, "similarity": round(sim, 4)}
            for item, sim in scores[:top_k]
            if sim > 0.01
        ]
        return candidates


# STAGE 6: Rule-based Filter (CC Part2 SFR 도메인 제약)

class RuleBasedFilter:
    """도메인 규칙으로 CC 후보 정제"""

    def _detect_domain(self, req: SecurityRequirement) -> list[str]:
        """요구사항 텍스트에서 도메인 탐지"""
        combined = f"{req.action} {req.object} {req.raw_text}".lower()
        domains = set()
        for keyword, domain in SECURITY_KEYWORD_DOMAIN_MAP.items():
            if keyword.lower() in combined:
                domains.add(domain)
        return list(domains)

    def filter(self, req: SecurityRequirement, candidates: list[dict]) -> list[dict]:
        """도메인 규칙에 맞는 후보만 통과"""
        domains = self._detect_domain(req)
        if not domains:
            # 도메인 미탐지 시 전체 후보 반환 (신뢰도 페널티)
            for c in candidates:
                c["similarity"] *= 0.7
            return candidates

        # 도메인에 해당하는 family 집합
        allowed_families = set()
        for domain in domains:
            allowed_families.update(DOMAIN_RULES.get(domain, []))

        filtered = [
            c for c in candidates
            if any(c["family"].startswith(fam) for fam in allowed_families)
        ]
        # 필터 결과 없으면 원본 반환
        return filtered if filtered else candidates


# STAGE 7: LLM Selector

class LLMSelector:
    """Anthropic API를 이용하여 최종 CC SFR 매핑 항목 선택"""

    def __init__(self):
        if ANTHROPIC_AVAILABLE:
            self.client = anthropic.Anthropic()
        else:
            self.client = None

    def _build_prompt(self, req: SecurityRequirement, candidates: list[dict]) -> str:
        cand_text = "\n".join([
            f"- {c['component']}: {c['description']} (유사도: {c['similarity']:.3f})"
            for c in candidates
        ])
        return f"""당신은 CC(Common Criteria) Part2 보안 기능 요구사항 전문가입니다.

아래 보안 요구사항에 가장 적합한 CC SFR 컴포넌트를 후보 목록에서 선택하세요.

[보안 요구사항]
- ID: {req.req_id}
- 주체(actor): {req.actor}
- 행위(action): {req.action}
- 객체(object): {req.object}
- 유형: {req.req_type}
- 원문: {req.raw_text}

[CC SFR 후보 목록]
{cand_text}

JSON 형식으로만 응답하세요 (마크다운 없이):
{{
  "selected_component": "<컴포넌트 ID>",
  "rationale": "<선택 이유 한국어 2문장>",
  "confidence": <0.0~1.0>
}}"""

    def select(self, req: SecurityRequirement, candidates: list[dict]) -> dict:
        """LLM으로 최적 CC SFR 컴포넌트 선택"""
        if not candidates:
            return {"selected_component": "NONE", "rationale": "적합한 후보 없음", "confidence": 0.0}

        if self.client:
            try:
                prompt = self._build_prompt(req, candidates)
                response = self.client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=512,
                    messages=[{"role": "user", "content": prompt}]
                )
                text = response.content[0].text.strip()
                # JSON 추출
                json_match = re.search(r'\{.*\}', text, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
            except Exception as e:
                print(f"  [LLM Selector] API 오류: {e} → fallback 사용")

        # Fallback: 가장 높은 유사도 후보 선택
        best = max(candidates, key=lambda c: c["similarity"])
        return {
            "selected_component": best["component"],
            "rationale": f"유사도 기반 자동 선택 (점수: {best['similarity']:.3f}). {best['description'][:60]}",
            "confidence": best["similarity"],
        }


# STAGE 8: Confidence Scoring

class ConfidenceScorer:
    """매핑 결과 신뢰도 산출"""

    def score(self, req: SecurityRequirement, selection: dict,
              candidates: list[dict]) -> float:
        base = float(selection.get("confidence", 0.5))

        # 요구사항 완성도 보정
        unknown_fields = sum(1 for f in [req.actor, req.action, req.object]
                             if f == "unknown")
        completeness_penalty = unknown_fields * 0.1

        # 후보 수 보정 (후보가 많을수록 선택 신뢰도 낮음)
        candidate_penalty = max(0, (len(candidates) - 3)) * 0.02

        # 유효성 보정
        validity_bonus = 0.1 if req.valid else -0.15

        final = base - completeness_penalty - candidate_penalty + validity_bonus
        return round(max(0.0, min(1.0, final)), 4)


# STAGE 9: Missing Requirement Detector

class MissingRequirementDetector:
    """CC Component Level 커버리지 기반 누락 요구사항 탐지"""

    # 보안 시스템의 필수 CC 클래스 기준
    ESSENTIAL_CLASSES = {
        "FAU": "감사(Audit) — 보안 이벤트 기록 및 검토",
        "FIA": "식별 및 인증(Identification & Authentication)",
        "FDP": "사용자 데이터 보호(User Data Protection)",
        "FCS": "암호화 지원(Cryptographic Support)",
        "FMT": "보안 관리(Security Management)",
        "FPT": "TSF 보호(Protection of TSF)",
    }
    RECOMMENDED_CLASSES = {
        "FCO": "통신 보안(Communication Security) — 부인방지",
        "FTA": "TOE 접근(TOE Access) — 세션 관리",
        "FRU": "자원 활용(Resource Utilization) — 가용성",
    }

    def detect(self, mappings: list[CCMapping]) -> tuple[list[dict], float]:
        """매핑된 CC 클래스 기준 커버리지 계산 및 누락 탐지"""
        covered_classes = {m.cc_class for m in mappings}
        covered_components = {m.cc_component for m in mappings}

        missing = []

        # 필수 클래스 누락 탐지
        for cc_class, desc in self.ESSENTIAL_CLASSES.items():
            if cc_class not in covered_classes:
                # 해당 클래스의 대표 컴포넌트 추출
                class_components = [
                    item["component"] for item in CC_SFR_DATABASE
                    if item["cc_class"] == cc_class
                ][:2]
                missing.append({
                    "type": "ESSENTIAL_MISSING",
                    "cc_class": cc_class,
                    "description": desc,
                    "suggested_components": class_components,
                    "recommendation": f"{desc} 관련 요구사항이 명시되지 않았습니다. "
                                     f"보안 평가를 위해 {', '.join(class_components)} 등의 SFR 충족을 검토하세요.",
                })

        # 권장 클래스 누락 탐지
        for cc_class, desc in self.RECOMMENDED_CLASSES.items():
            if cc_class not in covered_classes:
                class_components = [
                    item["component"] for item in CC_SFR_DATABASE
                    if item["cc_class"] == cc_class
                ][:1]
                missing.append({
                    "type": "RECOMMENDED_MISSING",
                    "cc_class": cc_class,
                    "description": desc,
                    "suggested_components": class_components,
                    "recommendation": f"권장 사항: {desc} 관련 요구사항 추가를 고려하세요.",
                })

        # 커버리지 계산
        total_essential = len(self.ESSENTIAL_CLASSES)
        covered_essential = sum(
            1 for c in self.ESSENTIAL_CLASSES if c in covered_classes
        )
        coverage = covered_essential / total_essential if total_essential > 0 else 0.0

        print(f"[Missing Detector] 필수 클래스 커버리지: {covered_essential}/{total_essential} "
              f"({coverage:.1%}), 누락 항목: {len(missing)}개")
        return missing, round(coverage, 4)


# STAGE 10: Structured Output

class StructuredOutputGenerator:
    """CC 매핑 매트릭스 + 누락 요구사항 설명 포함 최종 JSON 생성"""

    def generate(self, result: PipelineResult) -> dict:
        mapping_matrix = {}
        for m in result.cc_mappings:
            mapping_matrix[m.req_id] = {
                "cc_component": m.cc_component,
                "cc_family":    m.cc_family,
                "cc_class":     m.cc_class,
                "description":  m.description,
                "confidence":   m.confidence,
                "rationale":    m.rationale,
            }

        output = {
            "metadata": {
                "pipeline": "AI Harness Security Requirement Extractor v1.0",
                "total_requirements": len(result.requirements),
                "valid_requirements": sum(1 for r in result.requirements if r.valid),
                "total_mappings":     len(result.cc_mappings),
                "coverage_score":     result.coverage_score,
                "average_confidence": round(
                    sum(m.confidence for m in result.cc_mappings) / len(result.cc_mappings), 4
                ) if result.cc_mappings else 0.0,
            },
            "requirements": [
                {
                    "req_id":    r.req_id,
                    "actor":     r.actor,
                    "action":    r.action,
                    "object":    r.object,
                    "type":      r.req_type,
                    "valid":     r.valid,
                    "issues":    r.issues,
                    "raw_text":  r.raw_text,
                }
                for r in result.requirements
            ],
            "cc_mapping_matrix": mapping_matrix,
            "missing_requirements": result.missing_components,
            "explanation": {
                "coverage_analysis": (
                    f"추출된 {len(result.requirements)}개 보안 요구사항 중 "
                    f"{len(result.cc_mappings)}개가 CC Part2 SFR에 매핑되었습니다. "
                    f"필수 CC 클래스 커버리지는 {result.coverage_score:.1%}입니다."
                ),
                "missing_summary": (
                    f"총 {len(result.missing_components)}개 누락 항목이 탐지되었습니다. "
                    + ("필수 보안 통제 항목을 보완할 것을 권고합니다."
                       if any(m["type"] == "ESSENTIAL_MISSING"
                              for m in result.missing_components) else "")
                ),
                "quality_note": (
                    "actor/action/object가 'unknown'인 요구사항은 원문 명확화가 필요합니다."
                    if any(r.action == "unknown" or r.actor == "unknown"
                           for r in result.requirements) else
                    "모든 요구사항 구조가 완전히 식별되었습니다."
                ),
            },
        }
        return output


# 메인 파이프라인 오케스트레이터

class AIHarnessPipeline:
    """전체 파이프라인 실행"""

    def __init__(self):
        self.nlp_filter         = NLPFilter()
        self.intent_filter      = IntentFilter()
        self.req_extractor      = RequirementExtractor()
        self.req_validator      = RequirementValidator()
        self.cc_retriever       = CCCandidateRetriever()
        self.rule_filter        = RuleBasedFilter()
        self.llm_selector       = LLMSelector()
        self.confidence_scorer  = ConfidenceScorer()
        self.missing_detector   = MissingRequirementDetector()
        self.output_generator   = StructuredOutputGenerator()

    def run(self, stt_text: str) -> dict:
        print("=" * 60)
        print("AI Harness Security Requirement Pipeline 시작")
        print("=" * 60)

        # Stage 1: NLP Filter
        units = self.nlp_filter.process(stt_text)

        # Stage 2: Intent Filter
        classified = self.intent_filter.classify(units)

        # Stage 3: Requirement Extractor
        requirements = self.req_extractor.extract(classified)
        if not requirements:
            print("[Pipeline] 추출된 요구사항 없음. 종료.")
            return {"error": "요구사항을 추출할 수 없습니다.", "units": units}

        # Stage 4: Requirement Validator
        requirements = self.req_validator.validate(requirements)

        # Stages 5-8: CC 매핑 (Retrieval → Rule Filter → LLM Select → Scoring)
        cc_mappings = []
        print(f"\n[CC 매핑 시작] {len(requirements)}개 요구사항 처리...")
        for req in requirements:
            print(f"  처리 중: {req.req_id} | {req.action} / {req.object}")
            candidates   = self.cc_retriever.retrieve(req, top_k=5)
            candidates   = self.rule_filter.filter(req, candidates)
            selection    = self.llm_selector.select(req, candidates)
            confidence   = self.confidence_scorer.score(req, selection, candidates)

            # 선택된 컴포넌트 정보 조회
            selected_id  = selection.get("selected_component", "NONE")
            sfr_info     = next(
                (item for item in CC_SFR_DATABASE if item["component"] == selected_id),
                {"component": selected_id, "family": "UNKNOWN",
                 "cc_class": "UNKNOWN", "description": "알 수 없는 컴포넌트"}
            )
            mapping = CCMapping(
                req_id       = req.req_id,
                cc_component = selected_id,
                cc_family    = sfr_info["family"],
                cc_class     = sfr_info["cc_class"],
                description  = sfr_info["description"],
                confidence   = confidence,
                rationale    = selection.get("rationale", ""),
            )
            req.confidence = confidence
            cc_mappings.append(mapping)

        # Stage 9: Missing Requirement Detector
        missing, coverage = self.missing_detector.detect(cc_mappings)

        # 결과 조립
        result = PipelineResult(
            requirements       = requirements,
            cc_mappings        = cc_mappings,
            missing_components = missing,
            coverage_score     = coverage,
            mapping_matrix     = {},
        )
        # explanation은 output_generator에서 생성
        result.explanation = {}

        # Stage 10: Structured Output
        output = self.output_generator.generate(result)
        print("\n[Pipeline] 완료")
        return output


# 샘플 STT 입력 데이터

SAMPLE_STT_INPUT = """
보안 요구사항 회의록 - 2024년 시스템 보안 검토 회의

1. 시스템은 모든 사용자가 로그인 전에 반드시 인증을 수행해야 합니다.
2. 관리자는 시스템 내 모든 접근 이벤트에 대한 감사 로그를 기록해야 합니다.
3. 사용자 데이터는 AES-256 알고리즘으로 암호화해야 합니다.
4. 세션은 15분 이상 비활성 상태일 경우 자동으로 종료되어야 합니다.
5. 시스템은 역할 기반 접근 제어를 통해 사용자 권한을 관리해야 합니다.
6. 모든 통신은 TLS를 통해 암호화하여 전송해야 합니다.
7. 감사 로그는 무단 삭제로부터 보호되어야 합니다.
8. 관리자는 감사 기록을 열람할 수 있어야 합니다.
9. 시스템은 다중 인증을 지원해야 한다고 생각합니다. (의견)
10. 암호화 키는 안전하게 생성되고 관리되어야 합니다.
11. 사용자 식별 정보와 감사 이벤트를 연결하여 추적할 수 있어야 합니다.
12. 이 시스템은 의료 분야에서 사용될 예정입니다. (설명)
13. TSF는 자가 무결성 검사를 실행해야 합니다.
"""


# 메인 실행

def main():
    pipeline = AIHarnessPipeline()
    result   = pipeline.run(SAMPLE_STT_INPUT)

    # 결과 출력
    print("\n" + "=" * 60)
    print("최종 출력 결과 (CC 매핑 매트릭스 + 누락 요구사항)")
    print("=" * 60)
    print(json.dumps(result, ensure_ascii=False, indent=2))

    # 파일 저장
    output_path = "security_requirement_cc_mapping.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n[완료] 결과가 '{output_path}'에 저장되었습니다.")

    # 요약 출력
    meta = result.get("metadata", {})
    print("\n[요약]")
    print(f"  - 총 요구사항: {meta.get('total_requirements', 0)}개")
    print(f"  - 유효 요구사항: {meta.get('valid_requirements', 0)}개")
    print(f"  - CC 매핑 수: {meta.get('total_mappings', 0)}개")
    print(f"  - 평균 신뢰도: {meta.get('average_confidence', 0):.2%}")
    print(f"  - 필수 클래스 커버리지: {meta.get('coverage_score', 0):.1%}")
    missing = result.get("missing_requirements", [])
    essential_missing = [m for m in missing if m.get("type") == "ESSENTIAL_MISSING"]
    if essential_missing:
        print(f"  - 필수 누락 항목: {', '.join(m['cc_class'] for m in essential_missing)}")


if __name__ == "__main__":
    main()
