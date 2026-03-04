"""
STT-CC Mapping Tool
Layer 6 (Missing Analysis Layer)
- Requirements ↔ CC Part2 매핑 매트릭스 생성
- 누락 요구사항 감지 + 커버리지 계산
- Neo4j 저장 + CSV 출력
"""

import json
import csv
import os
from pathlib import Path
from datetime import datetime
from collections import defaultdict

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

# CC Part2 전체 표준 Family 목록 (기준 집합)
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



# 6. 누락 요구사항 감지 (Missing Analysis Layer)

class MissingAnalysisLayer:
    """Layer 6: 매핑 매트릭스 생성 → 커버리지 계산 → 누락 감지 → 저장/출력"""

    def __init__(self,
                 neo4j_uri: str = "bolt://localhost:7687",
                 neo4j_user: str = "neo4j",
                 neo4j_password: str = "password"):

        # Neo4j 드라이버 (선택적)
        try:
            from neo4j import GraphDatabase
            self.neo4j_driver = GraphDatabase.driver(
                neo4j_uri, auth=(neo4j_user, neo4j_password)
            )
            self.neo4j_driver.verify_connectivity()
            print(f"Layer 6. Neo4j 연결 완료: {neo4j_uri}")
        except Exception as e:
            print(f"Layer 6. Neo4j 연결 실패 (오프라인 모드): {e}")
            self.neo4j_driver = None

    # 로드 

    def _load_security_requirements(self, jsonld_path: str) -> list[dict]:
        """6-1. SecurityRequirementsList.jsonld 불러오기"""
        path = Path(jsonld_path)
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        requirements = data.get("@graph", [])
        print(f"Layer 6. SecurityRequirementsList 로드: {len(requirements)}개 요구사항")
        return requirements

    # 매트릭스 생성 

    def _build_mapping_matrix(self, requirements: list[dict]) -> list[dict]:
        """
        6-2. Requirements ↔ CC Part2 매핑 매트릭스 생성
        필드: requirement_id, intent_id, source_utterance,
              CC_family_id, missing_CC
        """
        # 실제 매핑된 CC 패밀리 집합
        mapped_families: set[str] = {
            req["CC_family_id"] for req in requirements if req.get("CC_family_id")
        }

        matrix_rows = []
        for req in requirements:
            cc_fid = req.get("CC_family_id", "UNKNOWN")
            # 해당 요구사항 행에서 누락된 패밀리 = 전체 - 이미 매핑된 패밀리
            # (행 단위 missing: 이 요구사항이 커버하지 않는 패밀리들)
            missing_cc = sorted(CC_PART2_ALL_FAMILIES.keys() - mapped_families)

            matrix_rows.append({
                "requirement_id":  req.get("requirement_id", ""),
                "intent_id":       req.get("intent_id", ""),
                "source_utterance": req.get("source_utterance", ""),
                "CC_family_id":    cc_fid,
                "CC_family_name":  CC_PART2_ALL_FAMILIES.get(cc_fid, "Unknown"),
                "missing_CC":      missing_cc,
                "gds_node_similarity": req.get("gds_node_similarity"),
            })

        print(f"Layer 6. 매핑 매트릭스 생성: {len(matrix_rows)}행")
        return matrix_rows

    # 커버리지 계산 

    def _calculate_coverage(self, matrix_rows: list[dict]) -> dict:
        """
        6-3. 커버리지 계산
        6-4. coverage_hit 반환
        """
        covered_families = {row["CC_family_id"] for row in matrix_rows
                            if row["CC_family_id"] not in ("UNKNOWN", "")}

        total_families   = len(CC_PART2_ALL_FAMILIES)
        covered_count    = len(covered_families & CC_PART2_ALL_FAMILIES.keys())
        missing_families = sorted(CC_PART2_ALL_FAMILIES.keys() - covered_families)

        coverage_hit = round(covered_count / total_families, 4) if total_families else 0.0

        coverage_report = {
            "total_cc_families":   total_families,
            "covered_families":    sorted(covered_families),
            "missing_families":    missing_families,
            "coverage_hit":        coverage_hit,
            "coverage_hit_pct":    f"{coverage_hit * 100:.1f}%",
        }

        print(f"Layer 6. CC Part2 커버리지: {coverage_report['coverage_hit_pct']} "
              f"({covered_count}/{total_families} families)")
        print(f"Layer 6. 누락 CC Family: {missing_families}")
        return coverage_report

    # Neo4j 저장 

    def _save_to_neo4j(self, matrix_rows: list[dict], coverage_report: dict) -> None:
        """
        6-5. Neo4j에 요구사항 ↔ CC Part2 관계,
             누락 요구사항, coverage_hit 저장
        """
        if not self.neo4j_driver:
            print("[Layer 6] Neo4j 미연결 → 저장 건너뜀")
            return

        with self.neo4j_driver.session() as session:
            # 누락 CC Family 노드 + MISSING 관계
            for row in matrix_rows:
                for missing_fid in row["missing_CC"]:
                    session.run("""
                        MERGE (r:Requirement {requirement_id: $req_id})
                        MERGE (f:CCFamily {family_id: $fid})
                          SET f.family_name = $fname, f.status = 'MISSING'
                        MERGE (r)-[:MISSING_CC]->(f)
                    """, req_id=row["requirement_id"],
                         fid=missing_fid,
                         fname=CC_PART2_ALL_FAMILIES.get(missing_fid, ""))

            # 커버리지 메타 노드
            session.run("""
                MERGE (c:CoverageReport {id: 'coverage_summary'})
                SET c.coverage_hit       = $hit,
                    c.covered_families   = $covered,
                    c.missing_families   = $missing,
                    c.total_cc_families  = $total,
                    c.updated_at         = $ts
            """,
                hit=coverage_report["coverage_hit"],
                covered=json.dumps(coverage_report["covered_families"]),
                missing=json.dumps(coverage_report["missing_families"]),
                total=coverage_report["total_cc_families"],
                ts=datetime.now().isoformat()
            )

        print("Layer 6. Neo4j 저장 완료 (누락 요구사항 + 커버리지)")

    # 출력 

    def _save_jsonld(self, matrix_rows: list[dict], coverage_report: dict) -> Path:
        """Requirements-CCPart2 Mapping Matrix.jsonld 저장"""
        jsonld = {
            "@context": {
                "@vocab":          "https://example.org/stt-cc#",
                "requirement_id":  "@id",
                "intent_id":       {"@type": "@id"},
                "CC_family_id":    "cc:familyId",
                "missing_CC":      "cc:missingFamily",
                "coverage_hit":    "cc:coverageHit",
            },
            "coverage_summary": coverage_report,
            "@graph":            matrix_rows,
            "created_at":        datetime.now().isoformat(),
        }
        out_path = OUTPUT_DIR / "Requirements-CCPart2_MappingMatrix.jsonld"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(jsonld, f, ensure_ascii=False, indent=2)
        print(f"[Layer 6] MappingMatrix.jsonld 저장: {out_path}")
        return out_path

    def _save_csv(self, matrix_rows: list[dict], coverage_report: dict) -> Path:
        """6-6. Requirements-CCPart2 Mapping Matrix.csv 출력"""
        out_path = OUTPUT_DIR / "Requirements-CCPart2_MappingMatrix.csv"

        fieldnames = [
            "requirement_id", "intent_id", "source_utterance",
            "CC_family_id", "CC_family_name", "missing_CC",
            "gds_node_similarity", "coverage_hit",
        ]

        with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in matrix_rows:
                writer.writerow({
                    "requirement_id":      row["requirement_id"],
                    "intent_id":           row["intent_id"],
                    "source_utterance":    row["source_utterance"],
                    "CC_family_id":        row["CC_family_id"],
                    "CC_family_name":      row["CC_family_name"],
                    "missing_CC":          "|".join(row["missing_CC"]),
                    "gds_node_similarity": row.get("gds_node_similarity", ""),
                    "coverage_hit":        coverage_report["coverage_hit"],
                })

        print(f"Layer 6. MappingMatrix.csv 저장: {out_path}")
        return out_path

    # 메인 실행 

    def process(self,
                jsonld_path: str = "output/SecurityRequirementsList.jsonld") -> dict:
        """Layer 6 전체 파이프라인"""

        # 6-1
        requirements = self._load_security_requirements(jsonld_path)

        # 6-2
        matrix_rows = self._build_mapping_matrix(requirements)

        # 6-3 / 6-4
        coverage_report = self._calculate_coverage(matrix_rows)

        # coverage_hit을 각 행에도 추가
        for row in matrix_rows:
            row["coverage_hit"] = coverage_report["coverage_hit"]

        # 6-5 Neo4j 저장
        self._save_to_neo4j(matrix_rows, coverage_report)

        # jsonld 중간 저장
        self._save_jsonld(matrix_rows, coverage_report)

        # 6-6 CSV 출력
        self._save_csv(matrix_rows, coverage_report)

        return {
            "matrix_rows":     matrix_rows,
            "coverage_report": coverage_report,
        }



# 실행 진입점 (Layer 6)

if __name__ == "__main__":
    layer6 = MissingAnalysisLayer(
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="password",
    )
    result = layer6.process(
        jsonld_path="output/SecurityRequirementsList.jsonld"
    )

    print("\n━━━ 최종 커버리지 요약 ━━━")
    cr = result["coverage_report"]
    print(f"  Coverage Hit : {cr['coverage_hit_pct']}")
    print(f"  Covered      : {cr['covered_families']}")
    print(f"  Missing      : {cr['missing_families']}")
    print("\n[완료] Layer 6 파이프라인 종료")
