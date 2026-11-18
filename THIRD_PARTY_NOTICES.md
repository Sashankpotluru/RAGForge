# Third-party software and models

The MIT license covers the original RAGForge code and synthetic fixture. Upstream software and model weights retain their own terms. Installed Python distributions carry their license metadata and bundled notices. The dependency graph and distribution hashes are recorded in [uv.lock](uv.lock).

## SciFact data

The public workbench redistributes the BEIR-formatted SciFact abstract collection
and a small set of training-split example claims. The abstract database is released
under ODC-By 1.0; claims and evidence annotations are CC BY 4.0, according to the
[original SciFact license](https://github.com/allenai/scifact/blob/master/LICENSE.md).
These data are separate from RAGForge's MIT-licensed code.

The workbench converts the official corpus to a browser JSON schema and publishes
derived MiniLM vectors. Original document IDs, titles and abstract text are
retained. The preparation manifest records archive, corpus and vector hashes.
This product includes data from the SciFact/S2ORC abstract database, made available
under the [Open Data Commons Attribution License](https://opendatacommons.org/licenses/by/1-0/).
Example claims retain the [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
attribution and license. No endorsement by the dataset authors is implied.

- David Wadden et al. (2020), [Fact or Fiction: Verifying Scientific Claims](https://aclanthology.org/2020.emnlp-main.609/), EMNLP.
- Nandan Thakur et al. (2021), [BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models](https://github.com/beir-cellar/beir), NeurIPS Datasets and Benchmarks.

## ArguAna benchmark data

The additional retrieval benchmark uses the BEIR distribution of ArguAna
Counterargs by Henning Wachsmuth, Shahbaz Syed and Benno Stein (2018).
[Paper](https://aclanthology.org/P18-1023/) ·
[Original dataset](https://webis.de/data/arguana-counterargs.html) ·
[BEIR dataset card](https://huggingface.co/datasets/BeIR/arguana)

The [original Zenodo record](https://zenodo.org/records/3973258) identifies the
original release as CC BY 4.0. The BEIR mirror labels its distribution
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). Redistributed
BEIR-derived query text and data adaptations in the ArguAna report retain the
CC BY-SA 4.0 notice. The repository's MIT license covers original program code,
not a replacement license for this third-party data.

RAGForge preserves official query/document IDs and query text, derives document
vectors, and adds rankings, measurements and provenance. Raw corpus downloads
and the ArguAna vector cache are obtained locally by the reproduction script;
they are not included in the GitHub Pages artifact. Attribution does not imply
endorsement by the authors, Webis, BEIR or the source debate platform.

## Software and model dependencies

PDF extraction uses PyMuPDF, distributed under GNU AGPL or a commercial license. Review the applicable terms for your distribution or hosted deployment. [PyMuPDF licensing](https://pymupdf.readthedocs.io/en/latest/about.html#license-and-copyright)

Qdrant and its Python client are separately distributed projects. The optional model downloads are external artifacts. Review each model card, revisions and upstream license before reuse. [Qdrant](https://github.com/qdrant/qdrant) · [Qdrant client](https://github.com/qdrant/qdrant-client) · [embedding fixture model](https://huggingface.co/sentence-transformers-testing/stsb-bert-tiny-safetensors) · [cross-encoder](https://huggingface.co/cross-encoder/ms-marco-TinyBERT-L2-v2)

The optional local generator uses separately downloaded Qwen2.5-0.5B-Instruct weights. [Pinned model and license](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/tree/7ae557604adf67be50417f59c2c2f167def9a775)

The GitHub Pages edition loads Transformers.js 3.8.1 and its ONNX Runtime from
jsDelivr when a visitor starts a neural operation. Model files are downloaded
separately from Hugging Face. These dependencies are not bundled into the static
site. Fixed revisions and download sizes are listed in the
[browser guide](docs/browser-demo.md#runtime-and-weights).

- [Transformers.js](https://github.com/huggingface/transformers.js/tree/3.8.1)
- [MiniLM embeddings](https://huggingface.co/Xenova/all-MiniLM-L6-v2)
- [MS MARCO cross-encoder](https://huggingface.co/Xenova/ms-marco-MiniLM-L-6-v2)
- [Flan-T5 Base](https://huggingface.co/Xenova/flan-t5-base)
