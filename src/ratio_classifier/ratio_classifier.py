import spacy

class RatioClassifier:
    ADJ_RATIO = 0.03
    ADV_RATIO = 0.04
    NUM_RATIO = 0.003

    def __init__(self):
        self.nlp = spacy.load("pt_core_news_sm")

    def extract_metrics(self, text: str) -> dict[int, int, int]:
        doc = self.nlp(text)
        total_tokens = len([token for token in doc if not token.is_punct and not token.is_space])

        num_adj = sum(1 for token in doc if token.pos_ == "ADJ")
        num_adv = sum(1 for token in doc if token.pos_ == "ADV")
        num_numbers = sum(1 for token in doc if token.pos_ == "NUM")

        return {
            "adj_ratio": num_adj / total_tokens,
            "adv_ratio": num_adv / total_tokens,
            "num_ratio": num_numbers / total_tokens
        }

    def news_classifier(self, text: str) -> str:
        metrics = self.extract_metrics(text=text)
        score = 0

        if metrics["adj_ratio"] >= self.ADJ_RATIO:
            score -= 1
        
        if metrics["adv_ratio"] >= self.ADV_RATIO:
            score -= 1

        if metrics["num_ratio"] >= self.NUM_RATIO:
            score += 1

        return "true" if score >= 0 else "fake"

ratio_classifier = RatioClassifier()
