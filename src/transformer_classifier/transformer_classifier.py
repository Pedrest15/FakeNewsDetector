import joblib
import torch
import math
from transformers import BertTokenizer, BertModel

EMBEDDING_ID = "neuralmind/bert-large-portuguese-cased" #335M
CLASSIFIER_FILE =  r"src\transformer_classifier\REGRAS_02_Bertimbau_RBFKernelSVM.joblib"

class TransformerClassifier:
    def __init__(self, embedding_id = EMBEDDING_ID, classifier_path = CLASSIFIER_FILE):
        self.tokenizer = BertTokenizer.from_pretrained(embedding_id)
        self.model = BertModel.from_pretrained(embedding_id)
        self.classifier = joblib.load(classifier_path)

    def get_sentence_embedding(x, tokenizer, model, pooling="special_token"):
        with torch.no_grad():
            token_limit = 512
            word_limit = int(token_limit*0.75) #more than that will be truncated
            words_list = x.split()
            #sentence_chunks_ammount = max([1,int(len(words_list)//word_limit)])
            
            overlap_fraction = 0.5
            non_overlapping_words_ammount = math.floor(word_limit*(1-overlap_fraction))
            sentence_chunks_number = math.floor(len(words_list)/(non_overlapping_words_ammount))
            sentence_chunks_number = max([sentence_chunks_number, 1])
            
            sentence_chunks = [words_list[non_overlapping_words_ammount*i:non_overlapping_words_ammount*i + word_limit] 
                            if i < sentence_chunks_number - 1 
                            else words_list[non_overlapping_words_ammount*i:] 
                            for i in range(sentence_chunks_number)]


            #sentence_chunks = [words_list[i*word_limit:(i+1)*word_limit] if i < sentence_chunks_ammount - 1 else words_list[i*word_limit:] for i in range(sentence_chunks_ammount)]
            sentence_chunks = [" ".join(sentence) for sentence in sentence_chunks]
            embeddings = []
            for chunk in sentence_chunks:  
                encoded_input = tokenizer(chunk, add_special_tokens=True, truncation=True, max_length=512, return_attention_mask=True)
                input_ids = torch.tensor(encoded_input["input_ids"]).unsqueeze(0)  # Batch size 1
                outputs = model(input_ids)

                last_hidden_states = outputs.last_hidden_state #(batch_size, input_len, embedding_size) But I need single vector for each sentence

                if pooling == "special_token":
                    embeddings.append(last_hidden_states[:, 0, :].cpu()[0]) #getting rid of batch axis since it's always just one element
                elif pooling == "max_pooling":
                    embeddings.append(torch.max(last_hidden_states, dim=1).values.cpu())
                elif pooling == "mean_pooling":
                    attention_mask = torch.tensor(encoded_input["attention_mask"])
                    attention_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_states.shape)
                    sum_embeddings = torch.sum(last_hidden_states * attention_mask_expanded, dim=1)
                    sum_mask = attention_mask.sum(dim=-1, keepdim=True)
                    sentence_embedding = sum_embeddings / sum_mask  # Normalize

                    embeddings.append(sentence_embedding.cpu())
                else:
                    raise ValueError(f"{pooling} is not a valid pooling technique!")

            return torch.mean(torch.stack(embeddings), dim=0)
    
    
    def news_classifier(self, sentence: str):
        sentence_embedding = self.get_sentence_embedding(sentence)
        prediction = self.classifier.predict(sentence_embedding.tolist())

        return "fake" if prediction == 1 else "true"

transformer_classifier = TransformerClassifier()