"""
STT-CC Mapping Tool
Layer 4 (Semantic Analysis) + Layer 5 (CC Part2 Mapping)
"""

import json
import os
import time
from pathlib import Path
from datetime import datetime

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)



# 4. 의미 분석 (Semantic Layer)
#    - GPT-4o로 발화를 REQUIRE / EXPLAIN / OPINION / DECISION 으로 분류
#    - IntentModel.jsonld 출력

class SemanticLayer:
    """Layer 4: GPT-4o Intent 분류 → IntentModel.jsonld"""

    # 4-3. Intent 분류 유형
    INTENT_TYPES = ["REQUIRE", "EXPLAIN", "OPINION", "DECISION"]

    _SYSTEM_PROMPT = """
당신은 정보보안 요구사항 분석 전문가입니다.
아래의 회의 발화(utterance)를 분석하여 각 발화의 의미 유형(intent_type)을 분류하십시오.

분류 기준:
- REQUIRE  : 보안 기능 구현, 정책 수립 등 명확한 '요구사항'을 표현하는 발화
- EXPLAIN  : 현황, 배경, 기술 개념 등을 '설명'하는 발화
- OPINION  : 개인적 의견, 제안, 선호 등 '단순 의견'을 표현하는 발화
- DECISION : 회의에서 확정된 '결정사항'을 표현하는 발화

출력 형식 (반드시 JSON 배열만 출력, 다른 텍스트 없음):
[
  {
    "intent_id": "INT-0001",
    "utterance_id": "SEG-0001",
    "intent_type": "REQUIRE",
    "source_utterance": "원문 발화",
    "confidence": 0.92,
    "reasoning": "분류 근거 한 줄"
  }
]
""".strip()

    def __init__(self, api_key: str | None = None):
        """4-1. OpenAI API(GPT-4o) 불러오기"""
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key or os.environ["OPENAI_API_KEY"])
            print("[Layer 4] OpenAI API 초기화 완료")
        except ImportError:
            raise ImportError("openai 패키지가 필요합니다: pip install openai")
        except KeyError:
            raise EnvironmentError("환경변수 OPENAI_API_KEY를 설정하세요.")

    # 내부 유틸 

    def _call_gpt(self, utterances: list[dict]) -> list[dict]:
        """GPT-4o 호출 (배치 단위)"""
        user_content = json.dumps(
            [{"utterance_id": u["utterance_id"], "text": u["text"]} for u in utterances],
            ensure_ascii=False, indent=2
        )

        response = self.client.chat.completions.create(
            model="gpt-4o",
            temperature=0.1,
            messages=[
                {"role": "system", "content": self._SYSTEM_PROMPT},
                {"role": "user",   "content": user_content},
            ],
        )
        raw = response.choices[0].message.content.strip()
        # JSON 펜스 제거
        if raw.startswith("```"):
            raw = re.sub(r"```[a-z]*\n?", "", raw).replace("```", "").strip()
        return json.loads(raw)

    # 메인 실행 

    def process(self, utterance_list: list[dict], batch_size: int = 20) -> list[dict]:
        """
        4-2. GPT-4o에 UtteranceList 입력
        4-3. 의미 구분 수행
        4-4. IntentModel 생성
        """
        intent_model = []
        intent_counter = 1

        for i in range(0, len(utterance_list), batch_size):
            batch = utterance_list[i:i + batch_size]
            try:
                results = self._call_gpt(batch)
                for item in results:
                    # intent_id 재부여 (GPT 출력값보다 순번 보장 우선)
                    item["intent_id"] = f"INT-{intent_counter:04d}"
                    intent_counter += 1
                    intent_model.append(item)
                time.sleep(0.5)     # Rate limit 여유
            except Exception as e:
                print(f"Layer 4. GPT 호출 오류 (배치 {i//batch_size+1}): {e}")
                # 오류 발화는 UNKNOWN으로 대체
                for u in batch:
                    intent_model.append({
                        "intent_id":        f"INT-{intent_counter:04d}",
                        "utterance_id":     u["utterance_id"],
                        "intent_type":      "UNKNOWN",
                        "source_utterance": u["text"],
                        "confidence":       0.0,
                        "reasoning":        f"GPT 오류: {e}",
                    })
                    intent_counter += 1

        print(f"Layer 4. 의미 분석 완료: {len(intent_model)}개 intent")
        return intent_model

    def save(self, intent_model: list[dict]) -> Path:
        """4-5. IntentModel.jsonld 출력하기"""
        jsonld = {
            "@context": {
                "@vocab":         "https://example.org/stt-cc#",
                "intent_id":      "@id",
                "intent_type":    "rdf:type",
                "source_utterance": "schema:text",
                "confidence":     "schema:value",
            },
            "@graph":     intent_model,
            "created_at": datetime.now().isoformat(),
        }
        out_path = OUTPUT_DIR / "IntentModel.jsonld"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(jsonld, f, ensure_ascii=False, indent=2)
        print(f"Layer 4. IntentModel.jsonld 저장: {out_path}")
        return out_path


# 5. CC 표준 매핑 (CC Part2 Mapping Layer)
#    - CCpart2Controls.json + IntentModel → SecurityRequirementsList.jsonld
#    - Neo4j 저장, GDS nodeSimilarity, score 병합

class CCMappingLayer:
    """Layer 5: CC Part2 매핑 → Neo4j → GDS → SecurityRequirementsList.jsonld"""

    _MAPPING_EXAMPLES = """
REQUIRE 발화 예시 → CC 매핑 예시 (Family Level):
- "사용자 인증을 강화해야 합니다" → FIA (Identification and Authentication)
- "감사 로그를 저장해야 합니다"   → FAU (Security Audit)
- "접근 통제 정책이 필요합니다"   → FDP (User Data Protection), FMT (Security Management)
- "암호화를 적용해야 합니다"      → FCS (Cryptographic Support)
- "세션 관리 기능이 필요합니다"   → FTA (TOE Access)
- "보안 패치 관리가 필요합니다"   → FPT (Protection of the TSF)
""".strip()

    _SYSTEM_PROMPT = """
당신은 Common Criteria (CC) Part 2 보안 기능 분류 전문가입니다.
아래의 보안 요구사항(REQUIRE 유형 발화)을 CC Part 2의 Family 수준에 매핑하십시오.

출력 형식 (반드시 JSON 배열만, 다른 텍스트 없음):
[
  {
    "requirement_id": "REQ-0001",
    "intent_id": "INT-0001",
    "source_utterance": "원문 발화",
    "CC_family_id": "FIA",
    "CC_family_name": "Identification and Authentication",
    "mapping_rationale": "매핑 근거 한 줄",
    "confidence": 0.88
  }
]
""".strip()

    def __init__(self, openai_api_key: str | None = None,
                 neo4j_uri: str = "bolt://localhost:7687",
                 neo4j_user: str = "neo4j",
                 neo4j_password: str = "password"):

        # 4-1 (재사용). OpenAI API
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=openai_api_key or os.environ["OPENAI_API_KEY"])
        except (ImportError, KeyError) as e:
            raise RuntimeError(f"OpenAI 초기화 실패: {e}")

        # 5-6. Neo4j 드라이버
        try:
            from neo4j import GraphDatabase
            self.neo4j_driver = GraphDatabase.driver(
                neo4j_uri, auth=(neo4j_user, neo4j_password)
            )
            self.neo4j_driver.verify_connectivity()
            print(f"Layer 5. Neo4j 연결 완료: {neo4j_uri}")
        except Exception as e:
            print(f"Layer 5. Neo4j 연결 실패 (오프라인 모드): {e}")
            self.neo4j_driver = None

        self.neo4j_uri = neo4j_uri

    # CCPart2Controls 로드 

    def _load_cc_controls(self, cc_path: str) -> dict:
        """5-1. CCpart2Controls.json 불러오기"""
        path = Path(cc_path)
        if not path.exists():
            print(f"[Layer 5] CCpart2Controls.json 없음 → 기본 패밀리 목록 사용")
            return self._default_cc_families()
        with open(path, encoding="utf-8") as f:
            controls = json.load(f)
        print(f"Layer 5. CC Part2 Controls 로드: {path}")
        return controls

    @staticmethod
    def _default_cc_families() -> dict:
        """CC Part2 표준 패밀리 기본값"""
        return {
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

    # GPT 매핑 

    def _call_gpt_mapping(self, require_intents: list[dict],
                          cc_controls: dict, req_counter_start: int) -> list[dict]:
        """5-2~5-4. GPT-4o로 Intent ⇔ CC Part2 매핑"""
        cc_summary = json.dumps(cc_controls, ensure_ascii=False, indent=2)
        user_content = (
            f"CC Part2 Families:\n{cc_summary}\n\n"
            f"매핑 예시:\n{self._MAPPING_EXAMPLES}\n\n"
            f"매핑 대상 요구사항:\n"
            + json.dumps(
                [{"intent_id": r["intent_id"], "source_utterance": r["source_utterance"]}
                 for r in require_intents],
                ensure_ascii=False, indent=2
            )
        )

        response = self.client.chat.completions.create(
            model="gpt-4o",
            temperature=0.1,
            messages=[
                {"role": "system", "content": self._SYSTEM_PROMPT},
                {"role": "user",   "content": user_content},
            ],
        )
        raw = response.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = re.sub(r"```[a-z]*\n?", "", raw).replace("```", "").strip()
        results = json.loads(raw)

        # requirement_id 순번 재부여
        for idx, item in enumerate(results):
            item["requirement_id"] = f"REQ-{req_counter_start + idx:04d}"
        return results

    # Neo4j 

    def _save_to_neo4j(self, requirements: list[dict]) -> None:
        """5-6. Neo4j에 요구사항 ↔ CC Part2 관계 저장"""
        if not self.neo4j_driver:
            print("Layer 5. Neo4j 미연결 → 저장 건너뜀")
            return

        cypher = """
        MERGE (r:Requirement {requirement_id: $requirement_id})
          SET r.source_utterance = $source_utterance,
              r.intent_id        = $intent_id,
              r.confidence       = $confidence
        MERGE (f:CCFamily {family_id: $CC_family_id})
          SET f.family_name = $CC_family_name
        MERGE (r)-[:MAPS_TO]->(f)
        """
        with self.neo4j_driver.session() as session:
            for req in requirements:
                session.run(cypher, **req)
        print(f"Layer 5. Neo4j 저장 완료: {len(requirements)}개 요구사항")

    # GDS 

    def _run_gds(self) -> list[dict]:
        """
        5-7. GDS graph projection 생성
        5-8. gds.nodeSimilarity 실행
        5-9. GDS_node_similarity (score) 반환
        """
        if not self.neo4j_driver:
            print("Layer 5. Neo4j 미연결 → GDS 건너뜀 (score=0.0)")
            return []

        with self.neo4j_driver.session() as session:
            # 기존 projection 정리
            session.run("""
                CALL gds.graph.exists('req-cc-graph') YIELD exists
                CALL apoc.do.when(exists,
                  'CALL gds.graph.drop($name) YIELD graphName RETURN graphName',
                  'RETURN null AS graphName',
                  {name: 'req-cc-graph'}
                ) YIELD value RETURN value
            """)

            # 5-7. Projection
            session.run("""
                CALL gds.graph.project(
                  'req-cc-graph',
                  ['Requirement', 'CCFamily'],
                  {MAPS_TO: {orientation: 'UNDIRECTED'}}
                )
            """)

            # 5-8. nodeSimilarity
            result = session.run("""
                CALL gds.nodeSimilarity.stream('req-cc-graph')
                YIELD node1, node2, similarity
                RETURN
                  gds.util.asNode(node1).requirement_id AS req1,
                  gds.util.asNode(node2).requirement_id AS req2,
                  similarity
                ORDER BY similarity DESC
                LIMIT 50
            """)
            scores = [dict(r) for r in result]

        print(f"Layer 5. GDS nodeSimilarity 완료: {len(scores)}개 쌍")
        return scores

    # 메인 실행 

    def process(self, intent_model: list[dict],
                cc_controls_path: str = "CCpart2Controls.json",
                batch_size: int = 20) -> list[dict]:
        """5-1 ~ 5-10. CC 매핑 전체 수행"""

        cc_controls = self._load_cc_controls(cc_controls_path)

        # REQUIRE 유형만 필터링
        require_intents = [i for i in intent_model if i.get("intent_type") == "REQUIRE"]
        print(f"Layer 5. REQUIRE 발화 수: {len(require_intents)}")

        requirements = []
        req_counter = 1
        for i in range(0, len(require_intents), batch_size):
            batch = require_intents[i:i + batch_size]
            try:
                mapped = self._call_gpt_mapping(batch, cc_controls, req_counter)
                requirements.extend(mapped)
                req_counter += len(mapped)
                time.sleep(0.5)
            except Exception as e:
                print(f"Layer 5. 매핑 오류 (배치 {i//batch_size+1}): {e}")

        # Neo4j 저장
        self._save_to_neo4j(requirements)

        # GDS
        gds_scores = self._run_gds()

        # 5-10. GDS score를 requirements에 병합
        score_map = {s["req1"]: s["similarity"] for s in gds_scores}
        for req in requirements:
            req["gds_node_similarity"] = score_map.get(req["requirement_id"], None)

        print(f"Layer 5. CC 매핑 완료: {len(requirements)}개 요구사항")
        return requirements, gds_scores

    def save(self, requirements: list[dict]) -> Path:
        """5-5. SecurityRequirementsList.jsonld 출력하기"""
        jsonld = {
            "@context": {
                "@vocab":           "https://example.org/stt-cc#",
                "requirement_id":   "@id",
                "intent_id":        {"@type": "@id"},
                "CC_family_id":     "cc:familyId",
                "source_utterance": "schema:text",
            },
            "@graph":     requirements,
            "created_at": datetime.now().isoformat(),
        }
        out_path = OUTPUT_DIR / "SecurityRequirementsList.jsonld"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(jsonld, f, ensure_ascii=False, indent=2)
        print(f"Layer 5. SecurityRequirementsList.jsonld 저장: {out_path}")
        return out_path



# 실행 진입점 (Layer 4-5)

import re

if __name__ == "__main__":
    # Layer 4. 의미 분석
    with open(OUTPUT_DIR / "UtteranceList.json", encoding="utf-8") as f:
        utterance_data = json.load(f)
    utterance_list = utterance_data["utterances"]

    layer4 = SemanticLayer()
    intent_model = layer4.process(utterance_list)
    layer4.save(intent_model)

    # Layer 5. CC 매핑
    layer5 = CCMappingLayer(
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="password",
    )
    requirements, gds_scores = layer5.process(intent_model)
    layer5.save(requirements)

    print("\n[완료] Layer 4-5 파이프라인 종료")
