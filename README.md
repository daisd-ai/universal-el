<p align="center">
    <img src="image.png" width="250" height="250">
</p>

# Universal Entity Linking

This repository contains the implementation of the entity linking pipeline proposed in our article [Universal entity linking](https://www.sciencedirect.com/science/article/pii/S0952197625021931). There are two versions available:
- **vLLM**: Optimized for speed but requires more VRAM.
- **HuggingFace**: More VRAM-efficient, suitable for consumer GPUs.


The vLLM solution is designed to be as fast as possible, which also means that it has higher VRAM usage. On the other hand HuggingFace solution is slower, but should fit into consumer focused GPUs.

## Disclaimer

- **vLLM**: Best for systems with high GPU resources (A100 or higher, with at least 40 GB of VRAM).
- **HuggingFace**: Suitable for consumer GPUs, but slower.

For lower-end GPUs, the vLLM version may require code adjustments to run efficiently.

## Installing

The pipeline uses the `Faiss` library (GPU version) and has been tested with Python 3.10 and A100 80GB GPU.

### vLLM version

1. Create a new Anaconda environment:

    ```bash
    conda create -n faiss_vllm python=3.10
    conda activate faiss_vllm
    ```

2. Install `faiss-gpu`:

    ```bash
    conda install pytorch::faiss-gpu
    ```

3. Install Python requirements:

    ```bash
    pip install -r reqs/requirements_vllm.txt
    ```

### HuggingFace version

1. Create a new Anaconda environment:

    ```bash
    conda create -n faiss_hf python=3.10
    conda activate faiss_hf
    ```

2. Install `faiss-gpu`:

    ```bash
    conda install pytorch::faiss-gpu
    ```

3. Install Python requirements:

    ```bash
    pip install -r reqs/requirements_hf.txt
    ```

## Faiss Wikidata index

The pipeline requires a Faiss Wikidata `IndexBinaryFlat`, downloadable via `wget`:

```bash
wget "https://box.pionier.net.pl/f/0f9c117b7bd24a0ab409/?dl=1" -O faiss_ubinary_wikidata_v2.index
```

The index is approximately 5 GB and contains around 40 million Wikidata entities. Refer to the publication for more details.

After successful download you can check if the file is not corrupted by running:

```bash
md5sum faiss_ubinary_wikidata_v2.index
```

it should return following hash: `82bada08ef48bc3ba15bf8c43d56b930`.

## Tests
To verify the installation, run the tests. Update the FAISS_INDEX_PATH in the test files with the path to your Faiss Wikidata index.

There are two test files, one for vLLM (`test_vllm_el.py`) based solution and the other for HuggingFace (`test_hf_el.py`). In both cases you need to specify path to the Faiss Wikidata index directly in given test file. 

```python
FAISS_INDEX_PATH = 'Faiss Wikidata index PATH GOES HERE'
```

After that you can run tests using specific conda environment while being in the project directory:

### vLLM test
```bash
conda activate faiss_vllm
python -m unittest tests/test_vllm_el.py
```

### HuggingFace test
```bash
conda activate faiss_hf
python -m unittest tests/test_hf_el.py
```

## Usage

The vLLM implementation provides the FastLinker, and HuggingFace offers the MemoryEfficientLinker. Both classes have identical methods and return the same results. Example usage with FastLinker:

Let's start with initializing the class.

```python
from src.vllm_el import FastLinker

el = FastLinker(faiss_index_path="PATH TO THE FAISS INDEX", gpu_memory_utilization=0.5, context_window_size=20)
```
In case of vLLM solution this can take awhile (on A100 80GB it takes about 45 seconds), HuggingFace solution initialization is faster and should not take more than few seconds.

There are other parameters of `FastLinker` class, but important are `gpu_memory_utilization` and `context_window_size`. 

The default value for `gpu_memory_utilization` is set to `0.5` and it works with A100 80GB, but it might be too low for other GPUs. In case of memory errors, try to increase it's value.

The `context_window_size` parameter decides how many words of each side of the entity should be taken into consideration while generating profile for the entity. More words should allow model to understand context better, at a cost of slower solution. The default is set to `20`.

The `FastLinker` provides four methods:
- `run_linking` - expects text and list of entities.
    ```python
    text = "Syracuse and Pitt in the # ACC ... its gon na be a long year for Maryland"
    entities = ["Syracuse", "Pitt", "ACC", "Maryland"]

    results = el.run_linking(text, entities)
    ```
- `run_chunks_linking` - expects list of texts and list of entities for each text.
    ```python
    texts = [
        "Syracuse and Pitt in the # ACC ... its gon na be a long year for Maryland",
        "MSU mens basketball signs Gary Harris . Womens basketball gets Mariah Harris . Harrises on Harrises on Harrises . # GottaHaveIt",
    ]
    entities = [
        ["Syracuse", "Pitt", "ACC", "Maryland"],
        ["MSU", "Harris"],
    ]

    results = el.run_chunks_linking(texts, entities)
    ```
- `run_positional_linking` - expects text, list of entities and list of indices. This is same implementation as `run_linking`, but instead of searching for positions of entities in text, expects to have it passed. It will be explained why later.
    ```python
    text = "Syracuse and Pitt in the # ACC ... its gon na be a long year for Maryland"
    entities = ["Syracuse", "Pitt", "ACC", "Maryland"]
    indices = [0, 13, 27, 65] #  only starting positions of entities

    results = el.run_positional_linking(text, entities, indices)
    ```
- `run_positional_chunks_linking` - same as `run_chunks_linking`, but also expects list of indices for each list of entities.
    ```python
    texts = [
        "Syracuse and Pitt in the # ACC ... its gon na be a long year for Maryland",
        "MSU mens basketball signs Gary Harris . Womens basketball gets Mariah Harris . Harrises on Harrises on Harrises . # GottaHaveIt",
    ]
    entities = [
        ["Syracuse", "Pitt", "ACC", "Maryland"],
        ["MSU", "Harris"],
    ]
    indices = [
        [0, 13, 27, 65],
        [0, 26],
    ]

    results = el.run_positional_chunks_linking(texts, entities)
    ```

Now let's explain why there are two, seemingly the same, pairs of methods.

In case of `run_linking` and `run_chunks_linking` there is internal method which search for all occurrences of entities. So for example in text:
```python
"Cat cat cat cat"
```
it will find all 4 occurrences of word `cat`. This also works in case of any characters next to the word, for example:

```python
"Cat1 cat's de'cat 6cat7"
```
Also method should return 4 occurrences. And this can be helpful in certain situations. The problem is that it will also work for words inside other words, for example:
```python
"Pitt is an acronym for Pittsburgh Panthers men's basketball."
```
Searching for word `Pitt` is going to return `Pitt` twice, one for `Pitt` and the other for `Pittsburgh`. 

That is why we implemented two other methods, `run_positional_linking` and `run_positional_chunks_linking`. They do not utilize this internal search method and only extract entities based on provided indices. These methods might be useful when using NER system that returns indices for found entities.

## License
The framework implementation, dataset for LLM fine tuning are released under `MIT` license. The `Tweeki_gold.jsonl`, `ISTEX_1000.jsonl`, `RSS_500.jsonl` and `anydef-orpo` LLM model are released under `Apache-2.0` license. The `reuters-128_wikidata.jsonl` is released under `AGPL-3.0` license.

## Citation
If you find our work useful, please cite using this BibTeX:
```bibtex
@article{RYNKIEWICZ2025112185,
    title = {Universal entity linking},
    journal = {Engineering Applications of Artificial Intelligence},
    volume = {161},
    pages = {112185},
    year = {2025},
    issn = {0952-1976},
    doi = {https://doi.org/10.1016/j.engappai.2025.112185},
    url = {https://www.sciencedirect.com/science/article/pii/S0952197625021931},
    author = {Adam Aron Rynkiewicz and Raul Palma and Piotr Formanowicz},
    keywords = {Entity linking, Large Language Model, Wikidata, Text embedding}
}
```
