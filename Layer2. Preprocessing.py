"""
Layer 2. Pre-processing 

ÁÖ¿ä ±â´É:
1. ÀâÀ½ Á¦°Å (ÇÊ·¯ ´Ü¾î, ¹İº¹, ºÒ¿ÏÀü ¹®Àå)
2. Á¤±ÔÈ­ (¸ÂÃã¹ı, ¶ç¾î¾²±â, ¹®Àå °æ°è)
3. ÆÄÆíÈ­µÈ ¹ßÈ­ º´ÇÕ
4. È­ÀÚ Á¤±ÔÈ­
"""

import re
import json
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import os
from collections import defaultdict


@dataclass
class Utterance:
    """¹ßÈ­ µ¥ÀÌÅÍ ±¸Á¶"""
    utterance_id: str
    speaker: str
    timestamp: str
    text: str
    original_text: str  # Á¤Á¦ Àü ¿øº»
    cleaned: bool = True
    merged_from: List[str] = None  # º´ÇÕµÈ °æ¿ì ¿øº» IDµé
    
    def __post_init__(self):
        if self.merged_from is None:
            self.merged_from = []
    
    def to_dict(self) -> Dict:
        return asdict(self)


class AdvancedPreprocessor:
    """°í±Ş ÀüÃ³¸® Å¬·¡½º - ÀÚ¿¬¾î È¸ÀÇ·Ï Ã³¸®"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.timestamp_pattern = re.compile(config['preprocessing']['timestamp_pattern'])
        self.speaker_pattern = re.compile(config['preprocessing']['speaker_pattern'])
        self.min_length = config['preprocessing']['min_utterance_length']
        
        # ÇÑ±¹¾î ÇÊ·¯ ´Ü¾î (Á¦°Å ´ë»ó)
        self.filler_words = [
            'À½', '¾î', '¾Æ', '±×', '¹¹', 'Àú', 'ÀÌÁ¦', '¸·', 
            '¾à°£', 'Á»', 'ÁøÂ¥', '±×³É', '¹º°¡', '±×·¯´Ï±î',
            '¿¡-', '¾î-', 'À½-', '¾Æ-', '±×-'
        ]
        
        # ¹İº¹ ÆĞÅÏ (¿¹: "±× ±× ±×°ÍÀº", "³×³×³×")
        self.repetition_pattern = re.compile(r'\b(\w{1,2})\s+\1(\s+\1)+\b')
        
        # ºÒ¿ÏÀü ¹®Àå ¸¶Ä¿
        self.incomplete_markers = ['...', '¡¦', '-', '--']
        
    def load_transcript(self, file_path: str) -> str:
        """Àü»ç ÆÄÀÏ ·Îµå"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def remove_noise(self, text: str) -> str:
        """ÀâÀ½ Á¦°Å"""
        cleaned = text
        
        # 1. ÇÊ·¯ ´Ü¾î Á¦°Å (¹®Àå ½ÃÀÛ À§Ä¡¿¡¼­¸¸)
        for filler in self.filler_words:
            # ¹®Àå ½ÃÀÛÀÌ³ª ½°Ç¥ ´ÙÀ½ÀÇ ÇÊ·¯¸¸ Á¦°Å
            cleaned = re.sub(rf'\b{filler}\s+', '', cleaned, flags=re.IGNORECASE)
        
        # 2. ¹İº¹ Á¦°Å (¿¹: "±× ±× ±×°ÍÀº" ¡æ "±×°ÍÀº")
        cleaned = self.repetition_pattern.sub(r'\1', cleaned)
        
        # 3. ¿¬¼ÓµÈ ½°Ç¥/¸¶Ä§Ç¥ Á¤¸®
        cleaned = re.sub(r'[,]{2,}', ',', cleaned)
        cleaned = re.sub(r'[.]{2,}', '.', cleaned)
        
        # 4. °ıÈ£ ¾ÈÀÇ ¹è°æ ¼ÒÀ½ Á¦°Å (¿¹: "(¿ôÀ½)", "(¹Ú¼ö)")
        cleaned = re.sub(r'\([^)]*¼ÒÀ½[^)]*\)', '', cleaned)
        cleaned = re.sub(r'\([^)]*¿ôÀ½[^)]*\)', '', cleaned)
        cleaned = re.sub(r'\([^)]*¹Ú¼ö[^)]*\)', '', cleaned)
        cleaned = re.sub(r'\([^)]*±âÄ§[^)]*\)', '', cleaned)
        
        # 5. ºÒÇÊ¿äÇÑ °ø¹é Á¤¸®
        cleaned = re.sub(r'\s+', ' ', cleaned)
        
        return cleaned.strip()
    
    def normalize_text(self, text: str) -> str:
        """ÅØ½ºÆ® Á¤±ÔÈ­"""
        normalized = text
        
        # 1. ¼ıÀÚ Á¤±ÔÈ­ (¿¹: "ÀÏÀÌ»ï" ¡æ À¯Áö, ÇÏÁö¸¸ "1 2 3" ÇüÅÂ´Â Á¤¸®)
        # ¿¬¼ÓµÈ ´ÜÀÏ ¼ıÀÚ¸¦ ´Ü¾î·Î º¯È¯ÇÏÁö ¾Ê°í ±×´ë·Î À¯Áö
        
        # 2. µû¿ÈÇ¥ ÅëÀÏ
        normalized = normalized.replace('"', '"').replace('"', '"')
        normalized = normalized.replace(''', "'").replace(''', "'")
        
        # 3. ¹°°áÇ¥¿Í ÇÏÀÌÇÂ Á¤±ÔÈ­
        normalized = normalized.replace('~', '¡­')
        normalized = normalized.replace('£­', '-')
        
        # 4. ¹®Àå ºÎÈ£ Á¤±ÔÈ­
        # ¶ç¾î¾²±â ¾øÀÌ ºÙÀº ¹®ÀåºÎÈ£ ¾Õ¿¡ °ø¹é Ãß°¡
        normalized = re.sub(r'([°¡-ÆRa-zA-Z0-9])([.!?])', r'\1 \2', normalized)
        
        # 5. ¿¬¼ÓµÈ °ø¹é Á¦°Å
        normalized = re.sub(r'\s+', ' ', normalized)
        
        return normalized.strip()
    
    def detect_sentence_boundary(self, text: str) -> bool:
        """¹®Àå ¿Ï°á¼º °Ë»ç"""
        # ¹®Àå Á¾°á ¾î¹Ì È®ÀÎ
        sentence_endings = ['´Ù.', '¿ä.', '±î.', 'ÁÒ.', '³×.', '±î?', '¿ä?', '³ª?', '´Ù!', '¿ä!']
        
        # ºÒ¿ÏÀü ¹®Àå ¸¶Ä¿ È®ÀÎ
        has_incomplete = any(marker in text for marker in self.incomplete_markers)
        
        # Á¾°á ¾î¹Ì·Î ³¡³ª´ÂÁö È®ÀÎ
        has_ending = any(text.rstrip().endswith(ending) for ending in sentence_endings)
        
        return has_ending and not has_incomplete
    
    def merge_fragments(self, utterances: List[Dict]) -> List[Dict]:
        """ÆÄÆíÈ­µÈ ¹ßÈ­ º´ÇÕ"""
        merged = []
        buffer = None
        
        for utt in utterances:
            text = utt['text']
            
            # ¿Ï°áµÈ ¹®ÀåÀÎÁö È®ÀÎ
            is_complete = self.detect_sentence_boundary(text)
            
            if buffer is None:
                # »õ·Î¿î ¹ßÈ­ ½ÃÀÛ
                if is_complete:
                    # ¿Ï°áµÈ ¹®ÀåÀÌ¸é ¹Ù·Î Ãß°¡
                    merged.append(utt)
                else:
                    # ºÒ¿ÏÀüÇÏ¸é ¹öÆÛ¿¡ ÀúÀå
                    buffer = utt.copy()
                    buffer['merged_from'] = [utt['utterance_id']]
            else:
                # ¹öÆÛ¿¡ ³»¿ëÀÌ ÀÖÀ¸¸é º´ÇÕ
                buffer['text'] += ' ' + text
                buffer['merged_from'].append(utt['utterance_id'])
                
                if is_complete:
                    # ¿Ï°áµÇ¸é ¹öÆÛ ³»¿ëÀ» Ãß°¡ÇÏ°í ÃÊ±âÈ­
                    merged.append(buffer)
                    buffer = None
        
        # ¸¶Áö¸· ¹öÆÛ°¡ ³²¾ÆÀÖÀ¸¸é Ãß°¡
        if buffer is not None:
            merged.append(buffer)
        
        return merged
    
    def normalize_speaker(self, speaker: str) -> str:
        """È­ÀÚ ÀÌ¸§ Á¤±ÔÈ­"""
        # °ø¹é Á¦°Å
        normalized = speaker.strip()
        
        # ¼ıÀÚ Á¦°Å (¿¹: "¹ßÇ¥ÀÚ1" ¡æ "¹ßÇ¥ÀÚ")
        # ÇÏÁö¸¸ ±¸ºĞÀÌ ÇÊ¿äÇÑ °æ¿ì¸¦ À§ÇØ ¼±ÅÃÀûÀ¸·Î¸¸ Àû¿ë
        # normalized = re.sub(r'\d+$', '', normalized)
        
        # °ıÈ£ Á¦°Å (¿¹: "¹ßÇ¥ÀÚ(±èÃ¶¼ö)" ¡æ "¹ßÇ¥ÀÚ")
        normalized = re.sub(r'\([^)]*\)', '', normalized).strip()
        
        return normalized
    
    def parse_utterances(self, transcript: str) -> List[Utterance]:
        """Àü»ç ÅØ½ºÆ®¸¦ ¹ßÈ­ ¸®½ºÆ®·Î ÆÄ½Ì"""
        utterances = []
        lines = transcript.strip().split('\n')
        
        utterance_counter = 1
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Å¸ÀÓ½ºÅÆÇÁ ÃßÃâ
            timestamp_match = self.timestamp_pattern.search(line)
            if not timestamp_match:
                continue
            
            timestamp = timestamp_match.group(1)
            remaining = line[timestamp_match.end():].strip()
            
            # È­ÀÚ ÃßÃâ
            speaker_match = self.speaker_pattern.match(remaining)
            if not speaker_match:
                continue
            
            speaker_raw = speaker_match.group(1).strip()
            speaker = self.normalize_speaker(speaker_raw)
            text_original = remaining[speaker_match.end():].strip()
            
            # ÃÖ¼Ò ±æÀÌ ÇÊÅÍ¸µ
            if len(text_original) < self.min_length:
                continue
            
            # ¹ßÈ­ °´Ã¼ »ı¼º
            utterance = Utterance(
                utterance_id=f"UTT_{utterance_counter:04d}",
                speaker=speaker,
                timestamp=timestamp,
                text=text_original,  # ¾ÆÁ÷ Á¤Á¦ Àü
                original_text=text_original,
                cleaned=False
            )
            
            utterances.append(utterance)
            utterance_counter += 1
        
        return utterances
    
    def clean_utterances(self, utterances: List[Utterance]) -> List[Utterance]:
        """¹ßÈ­ ¸®½ºÆ® Á¤Á¦"""
        cleaned = []
        
        for utt in utterances:
            # ÀâÀ½ Á¦°Å
            text_no_noise = self.remove_noise(utt.text)
            
            # Á¤±ÔÈ­
            text_normalized = self.normalize_text(text_no_noise)
            
            # ÃÖ¼Ò ±æÀÌ ÀçÈ®ÀÎ
            if len(text_normalized) >= self.min_length:
                utt.text = text_normalized
                utt.cleaned = True
                cleaned.append(utt)
        
        return cleaned
    
    def get_speaker_statistics(self, utterances: List[Utterance]) -> Dict:
        """È­ÀÚº° Åë°è"""
        stats = defaultdict(lambda: {'count': 0, 'total_chars': 0})
        
        for utt in utterances:
            stats[utt.speaker]['count'] += 1
            stats[utt.speaker]['total_chars'] += len(utt.text)
        
        # Æò±Õ ¹ßÈ­ ±æÀÌ °è»ê
        for speaker, data in stats.items():
            data['avg_length'] = data['total_chars'] / data['count']
        
        return dict(stats)
    
    def save_utterances(self, utterances: List[Utterance], output_path: str):
        """¹ßÈ­ ¸®½ºÆ®¸¦ JSONÀ¸·Î ÀúÀå"""
        speaker_stats = self.get_speaker_statistics(utterances)
        
        data = {
            "metadata": {
                "total_utterances": len(utterances),
                "processed_at": datetime.now().isoformat(),
                "speakers": list(set([u.speaker for u in utterances])),
                "speaker_statistics": speaker_stats,
                "preprocessing_config": {
                    "filler_words_removed": len(self.filler_words),
                    "min_utterance_length": self.min_length
                }
            },
            "utterances": [u.to_dict() for u in utterances]
        }
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def process(self, input_path: str, output_path: str, 
               enable_merge: bool = True) -> List[Utterance]:
        """ÀüÃ³¸® ÆÄÀÌÇÁ¶óÀÎ ½ÇÇà"""
        print(f"[ÀüÃ³¸®] ÀÔ·Â ÆÄÀÏ ·Îµå: {input_path}")
        transcript = self.load_transcript(input_path)
        
        print("[ÀüÃ³¸®] ¹ßÈ­ ÆÄ½Ì Áß...")
        utterances = self.parse_utterances(transcript)
        print(f"  ¡æ ÆÄ½ÌµÈ ¹ßÈ­ ¼ö: {len(utterances)}")
        
        print("[ÀüÃ³¸®] ÀâÀ½ Á¦°Å ¹× Á¤±ÔÈ­ Áß...")
        cleaned = self.clean_utterances(utterances)
        print(f"  ¡æ Á¤Á¦ ÈÄ ¹ßÈ­ ¼ö: {len(cleaned)}")
        
        if enable_merge:
            print("[ÀüÃ³¸®] ÆÄÆíÈ­µÈ ¹ßÈ­ º´ÇÕ Áß...")
            # Utterance °´Ã¼¸¦ dict·Î º¯È¯
            cleaned_dicts = [u.to_dict() for u in cleaned]
            merged_dicts = self.merge_fragments(cleaned_dicts)
            
            # dict¸¦ ´Ù½Ã Utterance °´Ã¼·Î º¯È¯
            merged = []
            for i, d in enumerate(merged_dicts):
                utt = Utterance(
                    utterance_id=d.get('utterance_id', f"UTT_{i+1:04d}"),
                    speaker=d['speaker'],
                    timestamp=d['timestamp'],
                    text=d['text'],
                    original_text=d.get('original_text', d['text']),
                    cleaned=d.get('cleaned', True),
                    merged_from=d.get('merged_from', [])
                )
                merged.append(utt)
            
            print(f"  ¡æ º´ÇÕ ÈÄ ¹ßÈ­ ¼ö: {len(merged)}")
            final = merged
        else:
            final = cleaned
        
        print(f"[ÀüÃ³¸®] °á°ú ÀúÀå: {output_path}")
        self.save_utterances(final, output_path)
        
        # Åë°è Ãâ·Â
        speaker_stats = self.get_speaker_statistics(final)
        print(f"\n[ÀüÃ³¸®] È­ÀÚº° Åë°è:")
        for speaker, stats in speaker_stats.items():
            print(f"  - {speaker}: {stats['count']}°³ ¹ßÈ­, Æò±Õ {stats['avg_length']:.1f}ÀÚ")
        
        return final


if __name__ == "__main__":
    import yaml
    
    # ¼³Á¤ ·Îµå
    with open('config/config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # ÀüÃ³¸® ½ÇÇà
    preprocessor = AdvancedPreprocessor(config)
    utterances = preprocessor.process(
        input_path='data/input/meeting_transcript_sample.txt',
        output_path='data/output/utterances.json',
        enable_merge=True  # ÆÄÆí º´ÇÕ È°¼ºÈ­
    )
    
    print(f"\n[¿Ï·á] ÃÑ {len(utterances)}°³ ¹ßÈ­ Ã³¸® ¿Ï·á")
    print(f"\n»ùÇÃ ¹ßÈ­:")
    for utt in utterances[:3]:
        print(f"  [{utt.speaker}] {utt.text[:60]}...")
        if utt.merged_from:
            print(f"    (º´ÇÕ: {len(utt.merged_from)}°³ ¹ßÈ­)")
