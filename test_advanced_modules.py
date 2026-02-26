"""
고급 전처리 및 NLP 모듈 통합 테스트 스크립트
"""

import yaml
import json
from src.preprocessing.advanced_preprocessor import AdvancedPreprocessor
from src.nlp.advanced_intent_classifier import AdvancedIntentClassifier


def compare_preprocessing():
    """전처리 결과 비교"""
    print("=" * 80)
    print("전처리 모듈 비교 테스트")
    print("=" * 80)
    
    with open('config/config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    preprocessor = AdvancedPreprocessor(config)
    
    # 자연스러운 회의록 처리 (병합 비활성화)
    print("\n[테스트 1] 자연스러운 회의록 - 병합 비활성화")
    print("-" * 80)
    utterances = preprocessor.process(
        input_path='data/input/natural_meeting_transcript.txt',
        output_path='data/output/utterances_natural_no_merge.json',
        enable_merge=False
    )
    
    # 샘플 출력
    print("\n정제 효과:")
    for i, utt in enumerate(utterances[:3]):
        print(f"\n{i+1}. [{utt.speaker}]")
        print(f"   원본: {utt.original_text}")
        print(f"   정제: {utt.text}")
    
    # 자연스러운 회의록 처리 (병합 활성화)
    print("\n\n[테스트 2] 자연스러운 회의록 - 병합 활성화")
    print("-" * 80)
    utterances_merged = preprocessor.process(
        input_path='data/input/natural_meeting_transcript.txt',
        output_path='data/output/utterances_natural_merged.json',
        enable_merge=True
    )
    
    print(f"\n병합 효과:")
    print(f"  - 병합 전: {len(utterances)}개 발화")
    print(f"  - 병합 후: {len(utterances_merged)}개 발화")
    
    # 병합된 발화 확인
    merged_count = sum(1 for utt in utterances_merged if len(utt.merged_from) > 0)
    print(f"  - 병합된 발화: {merged_count}개")


def compare_nlp():
    """NLP 모듈 비교"""
    print("\n\n" + "=" * 80)
    print("NLP 모듈 비교 테스트")
    print("=" * 80)
    
    with open('config/config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    classifier = AdvancedIntentClassifier(config)
    
    # 자연스러운 회의록 분석
    print("\n[분석] 자연스러운 회의록")
    print("-" * 80)
    intents = classifier.process(
        utterances_path='data/output/utterances_natural_no_merge.json',
        output_path='data/output/intents_natural_advanced.json'
    )
    
    # 상세 결과 출력
    print("\n\n상세 분석 결과 (상위 5개):")
    print("-" * 80)
    for i, intent in enumerate(intents[:5]):
        print(f"\n{i+1}. {intent.intent_type} (신뢰도: {intent.confidence:.2f})")
        if intent.sub_intent:
            print(f"   세부 의도: {intent.sub_intent}")
        print(f"   감정: {intent.sentiment}")
        print(f"   키워드: {', '.join(intent.keywords[:5])}")


def show_statistics():
    """통계 비교"""
    print("\n\n" + "=" * 80)
    print("전체 통계 비교")
    print("=" * 80)
    
    # 데이터 로드
    with open('data/output/utterances_natural_no_merge.json', 'r', encoding='utf-8') as f:
        utt_data = json.load(f)
    
    with open('data/output/intents_natural_advanced.json', 'r', encoding='utf-8') as f:
        intent_data = json.load(f)
    
    print("\n[전처리 통계]")
    print(f"  총 발화 수: {utt_data['metadata']['total_utterances']}")
    print(f"  화자 수: {len(utt_data['metadata']['speakers'])}")
    print(f"  화자 목록: {', '.join(utt_data['metadata']['speakers'])}")
    
    print("\n  화자별 발화 통계:")
    for speaker, stats in utt_data['metadata']['speaker_statistics'].items():
        print(f"    - {speaker}: {stats['count']}개, 평균 {stats['avg_length']:.1f}자")
    
    print("\n[NLP 분석 통계]")
    dist = intent_data['metadata']['distribution']
    
    print(f"  의도 분포:")
    for intent_type, count in dist['intent_types'].items():
        percentage = (count / intent_data['metadata']['total_intents']) * 100
        print(f"    - {intent_type}: {count}개 ({percentage:.1f}%)")
    
    print(f"\n  세부 의도 분포:")
    for sub_intent, count in dist['sub_intents'].items():
        print(f"    - {sub_intent}: {count}개")
    
    print(f"\n  감정 분포:")
    for sentiment, count in dist['sentiments'].items():
        percentage = (count / intent_data['metadata']['total_intents']) * 100
        print(f"    - {sentiment}: {count}개 ({percentage:.1f}%)")
    
    print(f"\n  상위 키워드 (Top 10):")
    for kw_data in intent_data['metadata']['top_keywords'][:10]:
        print(f"    - {kw_data['keyword']}: {kw_data['count']}회")


def main():
    """메인 함수"""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "고급 전처리 & NLP 모듈 통합 테스트" + " " * 24 + "║")
    print("╚" + "=" * 78 + "╝")
    
    try:
        # 전처리 비교
        compare_preprocessing()
        
        # NLP 비교
        compare_nlp()
        
        # 통계 비교
        show_statistics()
        
        print("\n\n" + "=" * 80)
        print("✅ 모든 테스트 완료!")
        print("=" * 80)
        
        print("\n📁 생성된 파일:")
        print("  - data/output/utterances_natural_no_merge.json")
        print("  - data/output/utterances_natural_merged.json")
        print("  - data/output/intents_natural_advanced.json")
        
    except Exception as e:
        print(f"\n❌ 오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
