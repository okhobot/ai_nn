from sentence_transformers import SentenceTransformer
import json
import os
import numpy as np


class MemoryModule:
    def __init__(self, data_path="config/mem_data.json"):
        """
        Initialize the memory module
        :param data_path: Path to the memory data file
        """
        self.data_path = data_path
        if not os.path.exists(self.data_path):
            with open(self.data_path, 'w', encoding='utf-8') as f:
                f.write("{\"text\":[], \"embed\":[]}")
        
        with open(self.data_path, encoding='utf-8') as f:
            self.mem_data = json.load(f)

        self.model = SentenceTransformer("google/embeddinggemma-300m")
            
    def request(self, query, res_count=1):
        """
        Request similar entries from memory
        :param query: Query text to find similar entries
        :param res_count: Number of results to return
        :return: List of similar text entries
        """
        res_count = min(res_count, len(self.mem_data["text"]))
        if res_count == 0: 
            return []
        
        similarities = self.model.similarity(
            self.model.encode_query(query), 
            np.array(self.mem_data["embed"], dtype=np.float32)
        )[0].tolist()
        
        res = []
        for i in range(res_count):
            m_index = similarities.index(max(similarities))
            res.append(self.mem_data["text"][m_index])
            similarities[m_index] = 0
            
        return res

    def save(self, text):
        """
        Save text to memory
        :param text: Text to save
        """
        emb = self.model.encode_query(text)
        self.mem_data["text"].append(text)
        self.mem_data["embed"].append(emb.tolist())
        
        with open(self.data_path, 'w', encoding='utf-8') as f:
            json.dump(self.mem_data, f)


# For backward compatibility when running as standalone
if __name__ == "__main__":
    mm = MemoryModule("config/mem_data.json")
    mm.save("нейросама - нейросеть-витубер, созданная vedal")