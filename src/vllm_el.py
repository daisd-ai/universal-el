import logging
from typing import NamedTuple

import faiss
from sentence_transformers import SentenceTransformer
from sentence_transformers.quantization import quantize_embeddings
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")


class FastLinker:
    """
    A class that represents a FastLinker for entity linking.

    Args:
        faiss_index_path (str): The file path to the Faiss index.
        top_k (int): The number of top-k entities to return. Default is 1.
        llm_model (str): The LLM model to use for entity linking. Default is "daisd-ai/anydef-orpo".
        tensor_parallel_size (int): The size of the tensor parallelism. Default is 1.
        gpu_memory_utilization (float): The GPU memory utilization. Default is 0.5.
        embedding_model (str): The embedding model to use for sentence representation. Default is "mixedbread-ai/mxbai-embed-large-v1".
        context_window_size (int): Number of words to consider of each side of the entity. Default is 20.
    """

    def __init__(
        self,
        faiss_index_path: str,
        top_k: int = 1,
        llm_model: str = "daisd-ai/anydef-orpo",
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.5,
        embedding_model: str = "mixedbread-ai/mxbai-embed-large-v1",
        context_window_size: int = 20,
    ) -> None:
        logging.debug("Initializing LLM model")
        self.llm = LLM(
            model=llm_model,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
        )
        self.sampling_params = SamplingParams(temperature=0, max_tokens=64)

        logging.debug("Initializing tokenizer")
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model)

        logging.debug("Initializing embedding model")
        self.embedding_model = SentenceTransformer(
            embedding_model,
            prompts={
                "retrieval": "Represent this sentence for searching relevant passages: ",
            },
            default_prompt_name="retrieval",
            device="cuda",
        )

        logging.debug("Loading index")
        self.faiss_index = faiss.read_index_binary(faiss_index_path)

        logging.debug("Transferring to GPU")
        gpu_index = faiss.index_cpu_to_all_gpus(self.faiss_index.index)
        self.faiss_index.own_fields = False
        self.faiss_index.index = gpu_index
        self.faiss_index.own_fields = True

        self.top_k = top_k

        self.context_window_size = context_window_size

        logging.info("Linker initialized")

    @staticmethod
    def _find_all(string: str, substring: str) -> list[int]:
        """
        Find all occurrences of a substring in a string.

        Args:
            string (str): The input string.
            substring (str): The target substring.

        Returns:
            list[int]: A list of indices.
        """
        indices = []
        index = string.find(substring)
        while index >= 0:
            indices.append(index)
            index = string.find(substring, index + 1)
        return indices

    def _get_context(self, text: str, entity: str, indicies: list[int]) -> list[NamedTuple]:
        """
        Get the context surrounding the given entity in the text.

        Args:
            text (str): The input text.
            entity (str): The entity to find the context for.
            indicies (list[int]): The indices of the entity occurrences in the text.

        Returns:
            list[NamedTuple]: A list of NamedTuples representing the context for each entity occurrence.
                Each NamedTuple contains the following fields:
                - left (str): The left context.
                - entity (str): The entity.
                - right (str): The right context.
                - start (int): The start index of the entity in the text.
                - end (int): The end index of the entity in the text.
        """
        Context = NamedTuple("Context", [("left", str), ("entity", str), ("right", str), ("start", int), ("end", int)])

        contexts = []
        for index in indicies:
            left = " ".join(text[:index].split()[-self.context_window_size :])
            right = " ".join(text[index + len(entity) :].split()[: self.context_window_size])
            contexts.append(Context(left, entity, right, index, index + len(entity)))

        return contexts

    def _extract_from_text(self, text: str, entities: list[str]) -> NamedTuple:
        """
        Extracts contexts from the given text for the specified entities.

        Args:
            text (str): The input text from which to extract contexts.
            entities (list[str]): The list of entities for which to extract contexts.

        Returns:
            NamedTuple: A named tuple containing the extracted contexts.
        """
        entities = list(dict.fromkeys(entities))
        entities = [entity.lower() for entity in entities]
        text = text.lower()

        contexts = []
        for entity in entities:
            indicies = self._find_all(text, entity)

            if len(indicies) == 0:
                raise ValueError(f"Entity '{entity}' not found in text")

            contexts.extend(self._get_context(text, entity, indicies))

        return contexts

    def _prepare_prompts(self, contexts: list[NamedTuple]) -> list[str]:
        """
        Prepare prompts for the given text and entities.

        Args:
            contexts (list): A list of contexts.

        Returns:
            list[str]: A list of prompts.
        """
        prompts = []
        for context in contexts:
            messages = [
                {
                    "role": "user",
                    "content": f"{context.left} {context.entity} {context.right}",
                },
                {"role": "assistant", "content": "I read this text"},
                {"role": "user", "content": f"What is a profile for entity {context.entity}?"},
            ]
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            prompts.append(prompt)

        return prompts

    def _run_inference(self, prompts: list[str]) -> list[str]:
        """
        Run inference on the given prompts.

        Args:
            prompts (list): A list of prompts.

        Returns:
            list: A list of outputs.
        """
        outputs = self.llm.generate(prompts, self.sampling_params)

        profiles = [output.outputs[0].text for output in outputs]

        return profiles

    def _run_embedding(self, profiles: list[str]) -> list[list[float]]:
        """
        Run embedding on the given profiles.

        Args:
            profiles (list): A list of profiles.

        Returns:
            list: A list of embeddings.
        """
        embeddings = self.embedding_model.encode(profiles)
        embeddings = quantize_embeddings(embeddings, precision="ubinary")

        return embeddings

    def _search_index(self, embeddings: list[list[float]], contexts: list[NamedTuple]) -> list[dict]:
        """
        Search the Faiss index for the given embeddings.

        Args:
            embeddings (list): A list of embeddings.
            contexts (list): A list of contexts.

        Returns:
            list: A list of search results.
        """
        scores, indices = self.faiss_index.search(embeddings, self.top_k)

        results = []
        for context, score, index in zip(contexts, scores, indices):
            results.append(
                {
                    "entity": context.entity,
                    "identifier": f"Q{index[0]}",
                    "score": float(score[0]),
                    "start": context.start,
                    "end": context.end,
                }
            )

        return results

    def _core_linking(self, contexts: list[NamedTuple]) -> list[dict]:
        """
        Performs core linking for the given contexts.

        Args:
            contexts (list[NamedTuple]): A list of NamedTuples representing the contexts.

        Returns:
            list[dict]: A list of dictionaries representing the results of core linking.
        """
        prompts = self._prepare_prompts(contexts)
        profiles = self._run_inference(prompts)
        embeddings = self._run_embedding(profiles)
        results = self._search_index(embeddings, contexts)

        return results

    @staticmethod
    def _match_pattern(contexts: list[list[NamedTuple]], results: list[dict]) -> list[list[dict]]:
        """
        Matches patterns in the given contexts with the provided results.

        Args:
            contexts (list[list[NamedTuple]]): A list of lists containing NamedTuples representing the contexts.
            results (list[dict]): A list of dictionaries representing the results.

        Returns:
            list[list[dict]]: A list of lists containing dictionaries representing the matched patterns.
        """
        res_iter = iter(results)
        results = [[next(res_iter) for _ in sublist] for sublist in contexts]
        return results

    def run_linking(self, text: str, entities: list[str]) -> list[dict]:
        """
        Run entity linking on the given text and entities.

        Args:
            text (str): The input text.
            entities (list): A list of entities to link.

        Returns:
            list[dict]: A list of linked entities.
        """
        contexts = self._extract_from_text(text, entities)
        results = self._core_linking(contexts)

        return results

    def run_chunks_linking(self, texts: list[str], entities: list[list[str]]) -> list[list[dict]]:
        """
        Run entity linking on the given texts and entities.

        Args:
            texts (list): A list of input texts.
            entities (list): A list of entities to link.

        Returns:
            list[list[dict]]: A list of linked entities.
        """
        contexts = []
        for text, ents in zip(texts, entities):
            contexts.append(self._extract_from_text(text, ents))

        contexts_flat_lst = sum(contexts, [])
        results = self._core_linking(contexts_flat_lst)
        results = self._match_pattern(contexts, results)

        return results

    def run_positional_linking(self, text: str, entities: list[str], indicies: list[int]) -> list[dict]:
        """
        Runs positional linking on the given text, entities, and indices.

        Args:
            text (str): The input text.
            entities (list[str]): A list of entity names.
            indicies (list[int]): A list of indices corresponding to begining of each entity.

        Returns:
            list[dict]: A list of results from the positional linking process.
        """
        contexts = []

        for entity, ids in zip(entities, indicies):
            contexts.extend(self._get_context(text, entity, [ids]))

        results = self._core_linking(contexts)

        return results

    def run_positional_chunks_linking(
        self, texts: list[str], entities: list[list[str]], indicies: list[list[int]]
    ) -> list[list[dict]]:
        """
        Runs positional chunks linking on the given texts, entities, and indices.

        Args:
            texts (list[str]): A list of texts.
            entities (list[list[str]]): A list of entity lists.
            indicies (list[list[int]]): A list of index lists. Each number in the list corresponds to the begining of an entity.

        Returns:
            list[list[dict]]: A list of lists of dictionaries representing the results.
        """
        contexts = []

        for text, ents, ids in zip(texts, entities, indicies):
            context = []

            for entity, idx in zip(ents, ids):
                context.extend(self._get_context(text, entity, [idx]))

            contexts.append(context)

        contexts_flat_lst = sum(contexts, [])
        results = self._core_linking(contexts_flat_lst)
        results = self._match_pattern(contexts, results)

        return results
