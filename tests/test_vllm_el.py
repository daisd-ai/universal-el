import json
import os
import unittest

from src.vllm_el import FastLinker

FAISS_INDEX_PATH = "Faiss Wikidata index PATH GOES HERE"
TEST_FILE_PATH = os.path.join("tests", "files", "tests_Tweeki_gold.jsonl")


class TestFastLinker(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.el = FastLinker(faiss_index_path=FAISS_INDEX_PATH)

        dataset = {}
        with open(TEST_FILE_PATH, "r") as f:
            for line in f:
                data = json.loads(line)
                dataset[data["id"]] = data

        cls.texts = []
        cls.entities = []
        cls.indicies = []
        for data in dataset.values():
            cls.texts.append(data["sentence"])
            cls.entities.append([data["sentence"][index[0] : index[1]] for index in data["index"]])
            cls.indicies.append([index[0] for index in data["index"]])

    def test_el(self):
        results = self.el.run_linking(self.texts[0], self.entities[0])
        self.assertEqual([i["entity"] for i in results], [i.lower() for i in self.entities[0]])

    def test_positional_el(self):
        results = self.el.run_positional_linking(self.texts[0], self.entities[0], self.indicies[0])
        self.assertEqual([i["entity"] for i in results], self.entities[0])

    def test_chunks_el(self):
        results = self.el.run_chunks_linking(self.texts, self.entities)
        self.assertEqual([i["entity"] for i in sum(results, [])], [i.lower() for i in sum(self.entities, [])])

    def test_positional_chunks_el(self):
        results = self.el.run_positional_chunks_linking(self.texts, self.entities, self.indicies)
        self.assertEqual([i["entity"] for i in sum(results, [])], sum(self.entities, []))


if __name__ == "__main__":
    unittest.main()
