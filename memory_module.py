from sentence_transformers import SentenceTransformer
import json
import os
import numpy as np

class MemoryModule:
    def __init__(self, data_path="./mem_data.json"):
        self.data_path = data_path
        if not os.path.exists(self.data_path):
            with open(self.data_path, 'w') as f:
                f.write("{\"text\":[], \"embed\":[]}")
        with open(self.data_path) as f:
            self.mem_data = json.load(f)
            print(self.mem_data)

        self.model = SentenceTransformer("google/embeddinggemma-300m")
            
    def request(self, query, res_count=1):
        res_count = min(res_count, len(self.mem_data["text"]))
        if res_count == 0: 
            return []
        similarities = self.model.similarity(self.model.encode_query(query), np.array(self.mem_data["embed"], dtype=np.float32))[0].tolist()
        #print(similarities)
        res = []
        for i in range(res_count):
            m_index = similarities.index(max(similarities))
            #print(m_index, max(similarities))
            res.append(self.mem_data["text"][m_index])
            similarities[m_index] = 0
        return res

    def save(self, text):
        emb = self.model.encode_query(text)
        self.mem_data["text"].append(text)
        #print(emb)
        self.mem_data["embed"].append(emb.tolist())
        with open(self.data_path, 'w') as f:
            json.dump(self.mem_data, f)

if __name__ == "__main__":
    mm = MemoryModule("config/mem_data.json")
    print(mm.request("что это"))
    #print(mm.request("кря"))
    #print(mm.request("тест"))