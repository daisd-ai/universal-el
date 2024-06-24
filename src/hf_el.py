import logging
from typing import NamedTuple

import faiss
import torch
from accelerate.utils import release_memory
from sentence_transformers import SentenceTransformer
from sentence_transformers.quantization import quantize_embeddings
from transformers import AutoTokenizer, pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")


class MemoryEfficientLinker:
    """
    A class that represents a MemoryEfficientLinker for entity linking.

    Args:
        faiss_index_path (str): The path to the Faiss index.
        top_k (int, optional): The number of top results to retrieve. Defaults to 1.
        device_map (str, optional): The device mapping. Defaults to "auto".
        torch_dtype (torch.dtype, optional): The torch data type. Defaults to torch.bfloat16.
        load_in_4bit (bool, optional): Whether to load the model in 4-bit mode. Defaults to False.
        llm_model (str, optional): The LLM model. Defaults to "arynkiewicz/anydef-orpo".
        embedding_model (str, optional): The embedding model. Defaults to "mixedbread-ai/mxbai-embed-large-v1".
        context_window_size (int): Number of words to consider of each side of the entity. Default is 20.
    """

    def __init__(
        self,
        faiss_index_path: str,
        top_k: int = 1,
        device_map: str = "auto",
        torch_dtype: torch.dtype = torch.bfloat16,
        load_in_4bit: bool = False,
        batch_size: int = 16,
        llm_model: str = "arynkiewicz/anydef-orpo",
        embedding_model: str = "mixedbread-ai/mxbai-embed-large-v1",
        context_window_size: int = 20,
    ) -> None:
        self.faiss_index_path = faiss_index_path
        self.top_k = top_k

        self.llm = llm_model
        self.device_map = device_map
        self.torch_dtype = torch_dtype
        self.load_in_4bit = load_in_4bit
        self.batch_size = batch_size

        self.tokenizer = AutoTokenizer.from_pretrained(llm_model)

        self.embedding_model = embedding_model

        self.context_window_size = context_window_size

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
        Runs inference on the given prompts using the text-generation pipeline.

        Args:
            prompts (list[str]): List of prompts to generate text from.

        Returns:
            list[str]: List of generated texts corresponding to each prompt.
        """
        pipe = pipeline(
            "text-generation",
            model=self.llm,
            device_map=self.device_map,
            torch_dtype=self.torch_dtype,
            model_kwargs={"load_in_4bit": self.load_in_4bit},
            batch_size=self.batch_size,
        )

        # with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
        output = pipe(prompts, max_new_tokens=128, return_full_text=False)

        release_memory(pipe)

        output = [out[0]["generated_text"] for out in output]

        return output

    def _run_embedding(self, profiles: list[str]) -> list[list[float]]:
        """
        Encodes the given profiles using a sentence embedding model.

        Args:
            profiles (list[str]): A list of profiles to be encoded.

        Returns:
            list[list[float]]: A list of encoded embeddings for each profile.
        """
        embedding_model = SentenceTransformer(
            self.embedding_model,
            prompts={
                "retrieval": "Represent this sentence for searching relevant passages: ",
            },
            default_prompt_name="retrieval",
            device="cuda",
        )

        embeddings = embedding_model.encode(profiles)
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
        faiss_index = faiss.read_index_binary(self.faiss_index_path)

        gpu_index = faiss.index_cpu_to_all_gpus(faiss_index.index)
        faiss_index.own_fields = False
        faiss_index.index = gpu_index
        faiss_index.own_fields = True

        scores, indices = faiss_index.search(embeddings, self.top_k)

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
