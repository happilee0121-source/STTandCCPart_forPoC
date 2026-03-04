"""
STT-CC Mapping Tool
Layer 1 (Input) + Layer 2 (Pre-processing) + Layer 3 (NLP)
"""

import json
import re
import os
from pathlib import Path
from datetime import datetime


# 공통 설정
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)



# 1. 입력 데이터 (Input Layer)
#    - TranscriptSTT.txt 파일을 읽어 raw 텍스트로 반환 

class InputLayer:
    """Layer 1: TranscriptSTT.txt 로드"""

    def __init__(self, filepath: str):
        self.filepath = Path(filepath)

    def load(self) -> str:
        """1-1. TranscriptSTT.txt 입력하기"""
        if not self.filepath.exists():
            raise FileNotFoundError(f"STT 파일을 찾을 수 없습니다: {self.filepath}")
        with open(self.filepath, encoding="utf-8") as f:
            raw = f.read()
        print(f"[Layer 1] STT 파일 로드 완료: {self.filepath} ({len(raw)} chars)")
        return raw


# 2. 전처리 (Pre-processing Layer)
#    - Kiwi 형태소 분석기로 잡음 제거 · 정규화 후 MinutesInput.json 출력

class PreprocessingLayer:
    """Layer 2: 잡음 제거 → 정규화 → MinutesInput.json"""

    # ClovaNote STT 라인 패턴: "화자1  00:00  발화 내용"
    _LINE_PATTERN = re.compile(
        r"^(?P<speaker>[^\t\n]+?)\s{2,}(?P<timestamp>\d{1,2}:\d{2}(?::\d{2})?)\s+(?P<text>.+)$"
    )

    def __init__(self):
        # 2-1. Kiwi 한국어 형태소 분석기 불러오기
        try:
            from kiwipiepy import Kiwi
            self.kiwi = Kiwi()
            print("[Layer 2] Kiwi 형태소 분석기 초기화 완료")
        except ImportError:
            raise ImportError("kiwipiepy 패키지가 필요합니다: pip install kiwipiepy")

    
    # 내부 유틸리티 

    def _remove_noise(self, text: str) -> str:
        """잡음 제거: 필러어, 반복어, 특수문자 정리"""
        # 구어체 필러어 제거
        fillers = r"\b(음+|어+|아+|그+|저+|뭐+|네+|예+|응+)\b"
        text = re.sub(fillers, "", text)
        # 반복 문자 정리 (e.g. ㅋㅋㅋ → ㅋ)
        text = re.sub(r"(.)\1{2,}", r"\1", text)
        # 특수문자 정리 (마침표·쉼표·물음표·느낌표 제외)
        text = re.sub(r"[^\w\s가-힣.,?!]", " ", text)
        # 연속 공백 정리
        text = re.sub(r" {2,}", " ", text).strip()
        return text

    def _normalize(self, text: str) -> str:
        """정규화: 전각 → 반각, 소문자 통일, 줄임말 복원 등"""
        # 전각 숫자/알파벳 → 반각
        text = text.translate(str.maketrans(
            "０１２３４５６７８９ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ",
            "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
        ))
        return text.strip()

    def _parse_lines(self, raw: str) -> list[dict]:
        """ClovaNote STT 포맷 파싱 → [{speaker, timestamp, text}, ...]"""
        segments = []
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            m = self._LINE_PATTERN.match(line)
            if m:
                segments.append({
                    "speaker":   m.group("speaker").strip(),
                    "timestamp": m.group("timestamp").strip(),
                    "text":      m.group("text").strip(),
                })
            else:
                # 패턴 불일치 → 직전 화자의 연속 발화로 처리
                if segments:
                    segments[-1]["text"] += " " + line
        return segments

    
    # 메인 실행 

    def process(self, raw: str) -> list[dict]:
        """
        2-2. 잡음 제거, 파편화, 정규화
        2-3. speaker, timestamp 포함해서 MinutesInput 만들기
        """
        segments = self._parse_lines(raw)

        minutes_input = []
        for idx, seg in enumerate(segments):
            cleaned = self._remove_noise(seg["text"])
            normalized = self._normalize(cleaned)
            if not normalized:          # 잡음만 있던 발화 제외
                continue
            minutes_input.append({
                "segment_id": f"SEG-{idx+1:04d}",
                "speaker":    seg["speaker"],
                "timestamp":  seg["timestamp"],
                "text":       normalized,
            })

        print(f"[Layer 2] 전처리 완료: {len(minutes_input)}개 세그먼트")
        return minutes_input

    def save(self, minutes_input: list[dict]) -> Path:
        """2-4. MinutesInput.json 출력하기"""
        out_path = OUTPUT_DIR / "MinutesInput.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(
                {"created_at": datetime.now().isoformat(), "segments": minutes_input},
                f, ensure_ascii=False, indent=2
            )
        print(f"[Layer 2] MinutesInput.json 저장: {out_path}")
        return out_path


# 3. 자연어 처리 (NLP Layer)
#    - Kiwi CoNg로 어절 분해 · 명사구 추출 · 의존 관계 분석
#    - utterance_id, speaker, timestamp 포함 → UtteranceList.json 출력

class NLPLayer:
    """Layer 3: Kiwi CoNg 기반 NLP → UtteranceList.json"""

    def __init__(self):
        try:
            from kiwipiepy import Kiwi
            from kiwipiepy.utils import Stopwords
            self.kiwi = Kiwi()
            self.stopwords = Stopwords()
            print("[Layer 3] Kiwi NLP 엔진 초기화 완료")
        except ImportError:
            raise ImportError("kiwipiepy 패키지가 필요합니다: pip install kiwipiepy")

    # 내부 유틸리티 

    def _extract_noun_phrases(self, text: str) -> list[str]:
        """3-2. Kiwi CoNg로 어절 분해 + 명사구 추출"""
        tokens = self.kiwi.tokenize(text, stopwords=self.stopwords)
        noun_phrases = []
        current_np = []

        for token in tokens:
            # NNG(일반명사), NNP(고유명사), SL(외래어) 태그를 명사구 구성 요소로 허용
            if token.tag in ("NNG", "NNP", "SL", "XR"):
                current_np.append(token.form)
            else:
                if current_np:
                    noun_phrases.append(" ".join(current_np))
                    current_np = []

        if current_np:
            noun_phrases.append(" ".join(current_np))

        return list(dict.fromkeys(noun_phrases))   # 순서 보존 중복 제거

    def _analyze_dependency(self, text: str) -> list[dict]:
        """3-3. Kiwi CoNg로 의존 관계 분석"""
        try:
            # Kiwi의 형태소 분석 결과를 간이 의존 구조로 변환
            tokens = self.kiwi.tokenize(text)
            dep_result = []
            for i, token in enumerate(tokens):
                dep_result.append({
                    "index":  i,
                    "form":   token.form,
                    "tag":    str(token.tag),
                    "start":  token.start,
                    "len":    token.len,
                })
            return dep_result
        except Exception as e:
            return [{"error": str(e)}]

    # 메인 실행 

    def process(self, minutes_input: list[dict]) -> list[dict]:
        """
        3-1. MinutesInput 입력
        3-2/3-3. 명사구 추출 + 의존 관계 분석
        3-4. utterance_id, speaker, timestamp 포함 UtteranceList 생성
        """
        utterance_list = []
        for seg in minutes_input:
            noun_phrases = self._extract_noun_phrases(seg["text"])
            dependency   = self._analyze_dependency(seg["text"])

            utterance_list.append({
                "utterance_id":  seg["segment_id"],
                "speaker":       seg["speaker"],
                "timestamp":     seg["timestamp"],
                "text":          seg["text"],
                "noun_phrases":  noun_phrases,
                "dependency":    dependency,
            })

        print(f"[Layer 3] NLP 처리 완료: {len(utterance_list)}개 발화")
        return utterance_list

    def save(self, utterance_list: list[dict]) -> Path:
        """3-5. UtteranceList.json 출력하기"""
        out_path = OUTPUT_DIR / "UtteranceList.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(
                {"created_at": datetime.now().isoformat(), "utterances": utterance_list},
                f, ensure_ascii=False, indent=2
            )
        print(f"[Layer 3] UtteranceList.json 저장: {out_path}")
        return out_path



# 실행 진입점 (Layer 1-3)

if __name__ == "__main__":
    import sys

    stt_path = sys.argv[1] if len(sys.argv) > 1 else "TranscriptSTT.txt"

    # Layer 1. 입력
    layer1 = InputLayer(stt_path)
    raw_text = layer1.load()

    # Layer 2. 전처리
    layer2 = PreprocessingLayer()
    minutes_input = layer2.process(raw_text)
    layer2.save(minutes_input)

    # Layer 3. 자연어처리 
    layer3 = NLPLayer()
    utterance_list = layer3.process(minutes_input)
    layer3.save(utterance_list)

    print("\n[완료] Layer 1-3 파이프라인 종료")
