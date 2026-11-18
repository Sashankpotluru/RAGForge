# Third-party software and models

The MIT license covers the original RAGForge code and synthetic fixture. Upstream software and model weights retain their own terms. Installed Python distributions carry their license metadata and bundled notices. The dependency graph and distribution hashes are recorded in [uv.lock](uv.lock).

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
