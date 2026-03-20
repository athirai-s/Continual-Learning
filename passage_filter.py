from typing import List

class PassageFilter:
    def __init__(self, min_passage_length: int = 100):
        self.min_length = min_passage_length
    
    def filter(self, passages: List[str]) -> List[str]:
        passages = self.deduplicate(passages)
        passages = self.remove_stubs(passages)
        return passages
    
    def deduplicate(self, passages: List[str]) -> List[str]:
        seen: set[str] = set()
        result: list[str] = []

        for p in passages:
            if p not in seen:
                seen.add(p)
                result.append(p)
        return result
    
    def remove_stubs(self, passages: List[str]) -> List[str]:
        return [
            p for p in passages 
            if len(p) >= self.min_length
        ]