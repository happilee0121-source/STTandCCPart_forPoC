"""
STT-CC Mapping Tool — Full Pipeline Orchestrator
레이어별 모듈을 순서대로 실행하는 메인 진입점
사용법:
    python main.py --stt TranscriptSTT.txt [--openai-key sk-...] [--neo4j-uri bolt://...]
환경변수로도 설정 가능:
    OPENAI_API_KEY, NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
"""

import argparse
import os
import sys
import json
import time
import re
import csv
import logging
from pathlib import Path
from datetime import datetime

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("stt_cc_pipeline.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

# CC Part2 전체 패밀리 기준 집합
CC_PART2_ALL_FAMILIES = {
    "FAU": "Security Audit",
    "FCO": "Communication",
    "FCS": "Cryptographic Support",
    "FDP": "User Data Protection",
    "FIA": "Identification and Authentication",
    "FMT": "Security Management",
    "FPR": "Privacy",
    "FPT": "Protection of the TSF",
    "FRU": "Resource Utilisation",
    "FTA": "TOE Access",
    "FTP": "Trusted Path/Channels",
}



# Layer 1 · 입력 데이터 (Input Layer)
class InputLayer:
    def __init__(self, filepath: str):
        self.filepath = Path(filepath)

    def load(self) -> str:
        """1-1. TranscriptSTT.txt 입력하기"""
        if not self.filepath.exists():
            raise FileNotFoundError(f"STT 파일 없음: {self.filepath}")
        raw = self.filepath.read_text(encoding="utf-8")
        log.info("[Layer 1] STT 파일 로드 완료 (%d chars)", len(raw))
        return raw



# Layer 2 · 전처리 (Pre-processing Layer)

class PreprocessingLayer:
    _LINE_RE = re.compile(
        r"^(?P<speaker>.+?)\s{2,}(?P<timestamp>\d{1,2}:\d{2}(?::\d{2})?)\s+(?P<text>.+)$"
    )
    _FILLERS = re.compile(r"\b(음+|어+|아+|그+|저+|뭐+|네+|예+|응+)\b")

    def __init__(self):
        """2-1. Kiwi 한국어 형태소 분석기 불러오기"""
        try:
            from kiwipiepy import Kiwi
            self.kiwi = Kiwi()
            log.info("[Layer 2] Kiwi 형태소 분석기 초기화 완료")
        except ImportError:
            raise ImportError("pip install kiwipiepy")

    def _remove_noise(self, text: str) -> str:
        """잡음 제거: 필러어, 반복 문자, 특수기호"""
        text = self._FILLERS.sub("", text)
        text = re.sub(r"(.)\1{2,}", r"\1", text)
        text = re.sub(r"[^\w\s가-힣.,?!]", " ", text)
        return re.sub(r" {2,}", " ", text).strip()

    def _normalize(self, text: str) -> str:
        """정규화: 전각→반각"""
        wide   = "０１２３４５６７８９ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ"
        narrow = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
        return text.translate(str.maketrans(wide, narrow)).strip()

    def _parse_lines(self, raw: str) -> list:
        """ClovaNote STT 포맷 파싱"""
        segs = []
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            m = self._LINE_RE.match(line)
            if m:
                segs.append({k: m.group(k).strip() for k in ("speaker", "timestamp", "text")})
            elif segs:
                segs[-1]["text"] += " " + line
        return segs

    def process(self, raw: str) -> list:
        """
        2-2. 잡음 제거, 파편화, 정규화
        2-3. speaker, timestamp 포함 MinutesInput 생성
        """
        minutes = []
        for idx, seg in enumerate(self._parse_lines(raw)):
            text = self._normalize(self._remove_noise(seg["text"]))
            if not text:
                continue
            minutes.append({
                "segment_id": f"SEG-{idx+1:04d}",
                "speaker":    seg["speaker"],
                "timestamp":  seg["timestamp"],
                "text":       text,
            })
        log.info("[Layer 2] 전처리 완료: %d 세그먼트", len(minutes))
        return minutes

    def save(self, minutes: list) -> Path:
        """2-4. MinutesInput.json 출력"""
        path = OUTPUT_DIR / "MinutesInput.json"
        path.write_text(
            json.dumps({"created_at": _now(), "segments": minutes},
                       ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        log.info("[Layer 2] MinutesInput.json 저장: %s", path)
        return path


# Layer 3 · 자연어 처리 (NLP Layer)

class NLPLayer:
    def __init__(self):
        try:
            from kiwipiepy import Kiwi
            from kiwipiepy.utils import Stopwords
            self.kiwi      = Kiwi()
            self.stopwords = Stopwords()
            log.info("[Layer 3] Kiwi NLP 엔진 준비 완료")
        except ImportError:
            raise ImportError("pip install kiwipiepy")

    def _extract_noun_phrases(self, text: str) -> list:
        """3-2. Kiwi CoNg 어절 분해 + 명사구 추출"""
        tokens = self.kiwi.tokenize(text, stopwords=self.stopwords)
        nps, cur = [], []
        for tok in tokens:
            if tok.tag in ("NNG", "NNP", "SL", "XR"):
                cur.append(tok.form)
            else:
                if cur:
                    nps.append(" ".join(cur))
                    cur = []
        if cur:
            nps.append(" ".join(cur))
        return list(dict.fromkeys(nps))

    def _dependency(self, text: str) -> list:
        """3-3. Kiwi CoNg 의존 관계 분석"""
        try:
            return [
                {"index": i, "form": t.form, "tag": str(t.tag),
                 "start": t.start, "len": t.len}
                for i, t in enumerate(self.kiwi.tokenize(text))
            ]
        except Exception as e:
            return [{"error": str(e)}]

    def process(self, minutes: list) -> list:
        """
        3-1. MinutesInput 입력
        3-2/3-3. 명사구 추출 + 의존 관계 분석
        3-4. utterance_id, speaker, timestamp 포함 UtteranceList 생성
        """
        result = []
        for seg in minutes:
            result.append({
                "utterance_id": seg["segment_id"],
                "speaker":      seg["speaker"],
                "timestamp":    seg["timestamp"],
                "text":         seg["text"],
                "noun_phrases": self._extract_noun_phrases(seg["text"]),
                "dependency":   self._dependency(seg["text"]),
            })
        log.info("[Layer 3] NLP 처리 완료: %d 발화", len(result))
        return result

    def save(self, utterances: list) -> Path:
        """3-5. UtteranceList.json 출력"""
        path = OUTPUT_DIR / "UtteranceList.json"
        path.write_text(
            json.dumps({"created_at": _now(), "utterances": utterances},
                       ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        log.info("[Layer 3] UtteranceList.json 저장: %s", path)
        return path


# Layer 4 · 의미 분석 (Semantic Layer)

class SemanticLayer:
    _SYS = """
당신은 정보보안 요구사항 분석 전문가입니다.
회의 발화(utterance)를 아래 기준으로 분류하십시오.

- REQUIRE  : 보안 기능·정책에 대한 명확한 요구사항
- EXPLAIN  : 현황·배경·개념 설명
- OPINION  : 개인 의견·선호·제안
- DECISION : 회의 확정 결정사항

반드시 JSON 배열만 출력하십시오. 다른 텍스트는 금지합니다.
[{"intent_id":"INT-0001","utterance_id":"...","intent_type":"REQUIRE",
  "source_utterance":"...","confidence":0.92,"reasoning":"..."}]
""".strip()

    def __init__(self, api_key=None):
        """4-1. OpenAI API(GPT-4o) 불러오기"""
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key or os.environ["OPENAI_API_KEY"])
            log.info("[Layer 4] OpenAI API 초기화 완료")
        except ImportError:
            raise ImportError("pip install openai")
        except KeyError:
            raise EnvironmentError("OPENAI_API_KEY 환경변수를 설정하세요.")

    def _call(self, batch: list) -> list:
        resp = self.client.chat.completions.create(
            model="gpt-4o",
            temperature=0.1,
            messages=[
                {"role": "system", "content": self._SYS},
                {"role": "user", "content": json.dumps(
                    [{"utterance_id": u["utterance_id"], "text": u["text"]}
                     for u in batch],
                    ensure_ascii=False, indent=2
                )},
            ],
        )
        return _parse_json(resp.choices[0].message.content)

    def process(self, utterances: list, batch_size: int = 20) -> list:
        """
        4-2. GPT-4o에 UtteranceList 입력
        4-3. REQUIRE / EXPLAIN / OPINION / DECISION 의미 구분
        4-4. intent_id, intent_type, source_utterance, confidence 포함 IntentModel 생성
        """
        intents, counter = [], 1
        for i in range(0, len(utterances), batch_size):
            batch = utterances[i:i + batch_size]
            try:
                items = self._call(batch)
                for item in items:
                    item["intent_id"] = f"INT-{counter:04d}"
                    counter += 1
                    intents.append(item)
                time.sleep(0.5)
            except Exception as e:
                log.warning("[Layer 4] GPT 오류 (batch %d): %s",
                            i // batch_size + 1, e)
                for u in batch:
                    intents.append({
                        "intent_id":        f"INT-{counter:04d}",
                        "utterance_id":     u["utterance_id"],
                        "intent_type":      "UNKNOWN",
                        "source_utterance": u["text"],
                        "confidence":       0.0,
                        "reasoning":        str(e),
                    })
                    counter += 1
        log.info("[Layer 4] 의미 분석 완료: %d intent", len(intents))
        return intents

    def save(self, intents: list) -> Path:
        """4-5. IntentModel.jsonld 출력"""
        path = OUTPUT_DIR / "IntentModel.jsonld"
        path.write_text(
            json.dumps({
                "@context": {
                    "@vocab":           "https://example.org/stt-cc#",
                    "intent_id":        "@id",
                    "intent_type":      "rdf:type",
                    "source_utterance": "schema:text",
                    "confidence":       "schema:value",
                },
                "@graph":     intents,
                "created_at": _now(),
            }, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        log.info("[Layer 4] IntentModel.jsonld 저장: %s", path)
        return path


# Layer 5 · CC Part2 매핑 (CC Part2 Mapping Layer)

class CCMappingLayer:
    _EXAMPLES = """
매핑 예시 (CC Family Level):
- "사용자 인증 강화"  → FIA (Identification and Authentication)
- "감사 로그 저장"    → FAU (Security Audit)
- "접근 통제 정책"    → FDP (User Data Protection), FMT (Security Management)
- "암호화 적용"       → FCS (Cryptographic Support)
- "세션 관리"         → FTA (TOE Access)
- "보안 패치 관리"    → FPT (Protection of the TSF)
""".strip()

    _SYS = """
당신은 Common Criteria Part 2 보안 기능 분류 전문가입니다.
보안 요구사항(REQUIRE 발화)을 CC Part 2 Family 수준에 매핑하십시오.

반드시 JSON 배열만 출력하십시오.
[{"requirement_id":"REQ-0001","intent_id":"INT-0001","source_utterance":"...",
  "CC_family_id":"FIA","CC_family_name":"Identification and Authentication",
  "mapping_rationale":"...","confidence":0.88}]
""".strip()

    def __init__(self, api_key=None,
                 neo4j_uri="bolt://localhost:7687",
                 neo4j_user="neo4j",
                 neo4j_password="password"):
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key or os.environ["OPENAI_API_KEY"])
        except (ImportError, KeyError) as e:
            raise RuntimeError(f"OpenAI 초기화 실패: {e}")
        self.neo4j_driver = _neo4j_connect(neo4j_uri, neo4j_user, neo4j_password, layer=5)

    def _load_cc_controls(self, path: str) -> dict:
        """5-1. CCpart2Controls.json 불러오기"""
        p = Path(path)
        if p.exists():
            log.info("[Layer 5] CC Controls 로드: %s", p)
            return json.loads(p.read_text(encoding="utf-8"))
        log.warning("[Layer 5] CCpart2Controls.json 없음 → 기본값 사용")
        return CC_PART2_ALL_FAMILIES

    def _call_mapping(self, batch: list, cc_controls: dict, start: int) -> list:
        """
        5-2. GPT-4o에 IntentModel 입력
        5-3. CC Part2 매핑 예시 입력
        5-4. Intent ↔ CC Part2 매핑 결과 생성
        """
        user_msg = (
            f"CC Part2 Families:\n{json.dumps(cc_controls, ensure_ascii=False, indent=2)}\n\n"
            f"{self._EXAMPLES}\n\n"
            f"매핑 대상:\n{json.dumps([{'intent_id': r['intent_id'], 'source_utterance': r['source_utterance']} for r in batch], ensure_ascii=False, indent=2)}"
        )
        resp = self.client.chat.completions.create(
            model="gpt-4o", temperature=0.1,
            messages=[
                {"role": "system", "content": self._SYS},
                {"role": "user",   "content": user_msg},
            ],
        )
        items = _parse_json(resp.choices[0].message.content)
        for idx, item in enumerate(items):
            item["requirement_id"] = f"REQ-{start + idx:04d}"
        return items

    def _save_neo4j(self, reqs: list):
        """5-6. Neo4j에 요구사항 ↔ CC Part2 관계 저장"""
        if not self.neo4j_driver:
            return
        with self.neo4j_driver.session() as s:
            for r in reqs:
                s.run("""
                    MERGE (req:Requirement {requirement_id: $rid})
                      SET req.source_utterance = $src, req.intent_id = $iid
                    MERGE (fam:CCFamily {family_id: $fid})
                      SET fam.family_name = $fname
                    MERGE (req)-[:MAPS_TO]->(fam)
                """, rid=r["requirement_id"],
                     src=r.get("source_utterance", ""),
                     iid=r.get("intent_id", ""),
                     fid=r.get("CC_family_id", ""),
                     fname=r.get("CC_family_name", ""))
        log.info("[Layer 5] Neo4j 저장: %d 요구사항", len(reqs))

    def _run_gds(self) -> list:
        """
        5-7. GDS graph projection 생성
        5-8. gds.nodeSimilarity 실행
        5-9. GDS_node_similarity (score) 반환
        """
        if not self.neo4j_driver:
            return []
        try:
            with self.neo4j_driver.session() as s:
                s.run("""
                    CALL gds.graph.project(
                      'req-cc-graph',
                      ['Requirement', 'CCFamily'],
                      {MAPS_TO: {orientation: 'UNDIRECTED'}}
                    )
                """)
                res = s.run("""
                    CALL gds.nodeSimilarity.stream('req-cc-graph')
                    YIELD node1, node2, similarity
                    RETURN gds.util.asNode(node1).requirement_id AS req1,
                           gds.util.asNode(node2).requirement_id AS req2,
                           similarity
                    ORDER BY similarity DESC LIMIT 50
                """)
                scores = [dict(r) for r in res]
            log.info("[Layer 5] GDS nodeSimilarity: %d 쌍", len(scores))
            return scores
        except Exception as e:
            log.warning("[Layer 5] GDS 오류: %s", e)
            return []

    def process(self, intents: list,
                cc_path: str = "CCpart2Controls.json",
                batch_size: int = 20):
        """5-1 ~ 5-10. CC 매핑 전체"""
        cc_controls     = self._load_cc_controls(cc_path)
        require_intents = [i for i in intents if i.get("intent_type") == "REQUIRE"]
        log.info("[Layer 5] REQUIRE 발화: %d건", len(require_intents))

        reqs, counter = [], 1
        for i in range(0, len(require_intents), batch_size):
            batch = require_intents[i:i + batch_size]
            try:
                mapped = self._call_mapping(batch, cc_controls, counter)
                reqs.extend(mapped)
                counter += len(mapped)
                time.sleep(0.5)
            except Exception as e:
                log.warning("[Layer 5] 매핑 오류 (batch %d): %s",
                            i // batch_size + 1, e)

        self._save_neo4j(reqs)
        gds_scores = self._run_gds()

        # 5-10. GDS score JSON-LD에 병합
        score_map = {s["req1"]: s["similarity"] for s in gds_scores}
        for r in reqs:
            r["gds_node_similarity"] = score_map.get(r["requirement_id"])

        log.info("[Layer 5] CC 매핑 완료: %d 요구사항", len(reqs))
        return reqs, gds_scores

    def save(self, reqs: list) -> Path:
        """5-5. requirement_id, intent_id, source_utterance, CC_family_id 포함
               SecurityRequirementsList.jsonld 출력"""
        path = OUTPUT_DIR / "SecurityRequirementsList.jsonld"
        path.write_text(
            json.dumps({
                "@context": {
                    "@vocab":           "https://example.org/stt-cc#",
                    "requirement_id":   "@id",
                    "intent_id":        {"@type": "@id"},
                    "CC_family_id":     "cc:familyId",
                    "source_utterance": "schema:text",
                },
                "@graph":     reqs,
                "created_at": _now(),
            }, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        log.info("[Layer 5] SecurityRequirementsList.jsonld 저장: %s", path)
        return path


# Layer 6 · 누락 요구사항 감지 (Missing Analysis Layer)

class MissingAnalysisLayer:
    def __init__(self, neo4j_uri="bolt://localhost:7687",
                 neo4j_user="neo4j", neo4j_password="password"):
        self.neo4j_driver = _neo4j_connect(neo4j_uri, neo4j_user, neo4j_password, layer=6)

    def _load(self, jsonld_path: str) -> list:
        """6-1. SecurityRequirementsList.jsonld 불러오기"""
        data = json.loads(Path(jsonld_path).read_text(encoding="utf-8"))
        reqs = data.get("@graph", [])
        log.info("[Layer 6] 요구사항 로드: %d건", len(reqs))
        return reqs

    def _build_matrix(self, reqs: list) -> list:
        """
        6-2. requirement_id, intent_id, source_utterance,
             CC_family_id, missing_CC 포함
             Requirements-CCPart2 Mapping Matrix 생성
        """
        covered       = {r["CC_family_id"] for r in reqs if r.get("CC_family_id")}
        missing_global = sorted(CC_PART2_ALL_FAMILIES.keys() - covered)
        rows = []
        for r in reqs:
            fid = r.get("CC_family_id", "UNKNOWN")
            rows.append({
                "requirement_id":      r.get("requirement_id", ""),
                "intent_id":           r.get("intent_id", ""),
                "source_utterance":    r.get("source_utterance", ""),
                "CC_family_id":        fid,
                "CC_family_name":      CC_PART2_ALL_FAMILIES.get(fid, "Unknown"),
                "missing_CC":          missing_global,
                "gds_node_similarity": r.get("gds_node_similarity"),
            })
        log.info("[Layer 6] 매핑 매트릭스: %d행", len(rows))
        return rows

    def _coverage(self, rows: list) -> dict:
        """
        6-3. Requirements-CCPart2 Mapping Matrix 바탕으로 커버리지 계산
        6-4. coverage_hit 반환
        """
        covered = {r["CC_family_id"] for r in rows} & CC_PART2_ALL_FAMILIES.keys()
        missing = sorted(CC_PART2_ALL_FAMILIES.keys() - covered)
        hit     = round(len(covered) / len(CC_PART2_ALL_FAMILIES), 4)
        report  = {
            "total_cc_families": len(CC_PART2_ALL_FAMILIES),
            "covered_families":  sorted(covered),
            "missing_families":  missing,
            "coverage_hit":      hit,
            "coverage_hit_pct":  f"{hit*100:.1f}%",
        }
        log.info("[Layer 6] Coverage: %s (%d/%d families)",
                 report["coverage_hit_pct"], len(covered), len(CC_PART2_ALL_FAMILIES))
        log.info("[Layer 6] Missing CC Families: %s", missing)
        return report

    def _save_neo4j(self, rows: list, coverage: dict):
        """
        6-5. Neo4j에 요구사항 ↔ CC Part2 관계,
             누락 요구사항, CC Part2 커버리지 저장
        """
        if not self.neo4j_driver:
            return
        with self.neo4j_driver.session() as s:
            for row in rows:
                for mfid in row["missing_CC"]:
                    s.run("""
                        MERGE (r:Requirement {requirement_id: $rid})
                        MERGE (f:CCFamily {family_id: $fid})
                          SET f.status = 'MISSING',
                              f.family_name = $fname
                        MERGE (r)-[:MISSING_CC]->(f)
                    """, rid=row["requirement_id"],
                         fid=mfid,
                         fname=CC_PART2_ALL_FAMILIES.get(mfid, ""))
            s.run("""
                MERGE (c:CoverageReport {id: 'coverage_summary'})
                SET c.coverage_hit     = $hit,
                    c.covered_families = $covered,
                    c.missing_families = $missing,
                    c.updated_at       = $ts
            """, hit=coverage["coverage_hit"],
                 covered=json.dumps(coverage["covered_families"]),
                 missing=json.dumps(coverage["missing_families"]),
                 ts=_now())
        log.info("[Layer 6] Neo4j 저장 완료 (누락 + 커버리지)")

    def _save_jsonld(self, rows: list, coverage: dict) -> Path:
        """Requirements-CCPart2 Mapping Matrix.jsonld 저장"""
        path = OUTPUT_DIR / "Requirements-CCPart2_MappingMatrix.jsonld"
        path.write_text(
            json.dumps({
                "@context": {
                    "@vocab":           "https://example.org/stt-cc#",
                    "requirement_id":   "@id",
                    "intent_id":        {"@type": "@id"},
                    "CC_family_id":     "cc:familyId",
                    "missing_CC":       "cc:missingFamily",
                    "coverage_hit":     "cc:coverageHit",
                },
                "coverage_summary": coverage,
                "@graph":           rows,
                "created_at":       _now(),
            }, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        log.info("[Layer 6] MappingMatrix.jsonld 저장: %s", path)
        return path

    def _save_csv(self, rows: list, coverage: dict) -> Path:
        """6-6. Requirements-CCPart2 Mapping Matrix.csv 출력"""
        path   = OUTPUT_DIR / "Requirements-CCPart2_MappingMatrix.csv"
        fields = [
            "requirement_id", "intent_id", "source_utterance",
            "CC_family_id", "CC_family_name", "missing_CC",
            "gds_node_similarity", "coverage_hit",
        ]
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for row in rows:
                w.writerow({
                    "requirement_id":      row["requirement_id"],
                    "intent_id":           row["intent_id"],
                    "source_utterance":    row["source_utterance"],
                    "CC_family_id":        row["CC_family_id"],
                    "CC_family_name":      row["CC_family_name"],
                    "missing_CC":          "|".join(row.get("missing_CC", [])),
                    "gds_node_similarity": row.get("gds_node_similarity", ""),
                    "coverage_hit":        coverage["coverage_hit"],
                })
        log.info("[Layer 6] MappingMatrix.csv 저장: %s", path)
        return path

    def process(self, jsonld_path: str = "output/SecurityRequirementsList.jsonld") -> dict:
        """Layer 6 전체 파이프라인"""
        reqs     = self._load(jsonld_path)
        rows     = self._build_matrix(reqs)
        coverage = self._coverage(rows)
        for row in rows:
            row["coverage_hit"] = coverage["coverage_hit"]
        self._save_neo4j(rows, coverage)
        self._save_jsonld(rows, coverage)
        self._save_csv(rows, coverage)
        return {"matrix_rows": rows, "coverage_report": coverage}



# 공통 유틸리티

def _now() -> str:
    return datetime.now().isoformat()


def _parse_json(text: str) -> list:
    """GPT 응답 JSON 파싱 (마크다운 펜스 제거)"""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"```[a-z]*\n?", "", text).replace("```", "").strip()
    return json.loads(text)


def _neo4j_connect(uri: str, user: str, password: str, layer: int):
    """Neo4j 드라이버 연결 (실패 시 None 반환 — 오프라인 모드)"""
    try:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver(uri, auth=(user, password))
        driver.verify_connectivity()
        log.info("[Layer %d] Neo4j 연결: %s", layer, uri)
        return driver
    except Exception as e:
        log.warning("[Layer %d] Neo4j 연결 실패 (오프라인 모드): %s", layer, e)
        return None


# CLI 진입점

def parse_args():
    p = argparse.ArgumentParser(description="STT-CC Mapping Tool Full Pipeline")
    p.add_argument("--stt",         default="TranscriptSTT.txt",
                   help="STT 입력 파일 경로")
    p.add_argument("--cc-controls", default="CCpart2Controls.json",
                   help="CC Part2 Controls JSON 경로")
    p.add_argument("--openai-key",  default=None,
                   help="OpenAI API Key (없으면 OPENAI_API_KEY 환경변수 사용)")
    p.add_argument("--neo4j-uri",   default=os.getenv("NEO4J_URI",      "bolt://localhost:7687"))
    p.add_argument("--neo4j-user",  default=os.getenv("NEO4J_USER",     "neo4j"))
    p.add_argument("--neo4j-pass",  default=os.getenv("NEO4J_PASSWORD", "password"))
    p.add_argument("--skip-layers", nargs="*", type=int, default=[],
                   help="건너뛸 레이어 번호 (캐시 사용, 예: --skip-layers 4 5)")
    return p.parse_args()


def main():
    args  = parse_args()
    skip  = set(args.skip_layers)
    n4j   = (args.neo4j_uri, args.neo4j_user, args.neo4j_pass)
    t0    = time.time()

    log.info("━━━ STT-CC Mapping Tool 파이프라인 시작 ━━━")

    # Layer 1  입력 
    raw_text = InputLayer(args.stt).load()

    # Layer 2  전처리 
    if 2 not in skip:
        layer2        = PreprocessingLayer()
        minutes_input = layer2.process(raw_text)
        layer2.save(minutes_input)
    else:
        minutes_input = json.loads(
            (OUTPUT_DIR / "MinutesInput.json").read_text(encoding="utf-8")
        )["segments"]
        log.info("[Layer 2] 건너뜀 → 캐시 로드")

    # Layer 3 NLP 
    if 3 not in skip:
        layer3         = NLPLayer()
        utterance_list = layer3.process(minutes_input)
        layer3.save(utterance_list)
    else:
        utterance_list = json.loads(
            (OUTPUT_DIR / "UtteranceList.json").read_text(encoding="utf-8")
        )["utterances"]
        log.info("[Layer 3] 건너뜀 → 캐시 로드")

    # Layer 4 의미 분석 
    if 4 not in skip:
        layer4       = SemanticLayer(api_key=args.openai_key)
        intent_model = layer4.process(utterance_list)
        layer4.save(intent_model)
    else:
        intent_model = json.loads(
            (OUTPUT_DIR / "IntentModel.jsonld").read_text(encoding="utf-8")
        )["@graph"]
        log.info("[Layer 4] 건너뜀 → 캐시 로드")

    # Layer 5 CC 매핑 
    if 5 not in skip:
        layer5 = CCMappingLayer(
            api_key=args.openai_key,
            neo4j_uri=n4j[0], neo4j_user=n4j[1], neo4j_password=n4j[2]
        )
        requirements, gds_scores = layer5.process(intent_model, cc_path=args.cc_controls)
        layer5.save(requirements)
    else:
        requirements = json.loads(
            (OUTPUT_DIR / "SecurityRequirementsList.jsonld").read_text(encoding="utf-8")
        )["@graph"]
        log.info("[Layer 5] 건너뜀 → 캐시 로드")

    # Layer 6  누락 분석 
    if 6 not in skip:
        layer6 = MissingAnalysisLayer(*n4j)
        result = layer6.process(str(OUTPUT_DIR / "SecurityRequirementsList.jsonld"))

        cr = result["coverage_report"]
        log.info("━━━ 최종 커버리지 요약 ━━━")
        log.info("  Coverage Hit : %s", cr["coverage_hit_pct"])
        log.info("  Covered      : %s", cr["covered_families"])
        log.info("  Missing      : %s", cr["missing_families"])

    log.info("━━━ 전체 파이프라인 완료 (%.1fs) ━━━", time.time() - t0)
    log.info("출력 디렉토리: %s", OUTPUT_DIR.resolve())


if __name__ == "__main__":
    main()
