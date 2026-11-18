# Third-party software and models

The MIT license covers the original RAGForge code and synthetic fixture. Upstream software and model weights retain their own terms. Installed Python distributions carry their license metadata and bundled notices. The dependency graph and distribution hashes are recorded in [uv.lock](uv.lock).

PDF extraction uses PyMuPDF, distributed under GNU AGPL or a commercial license. Review the applicable terms for your distribution or hosted deployment. [PyMuPDF licensing](https://pymupdf.readthedocs.io/en/latest/about.html#license-and-copyright)

Qdrant and its Python client are separately distributed projects. The optional model downloads are external artifacts. Review each model card, revisions and upstream license before reuse. [Qdrant](https://github.com/qdrant/qdrant) · [Qdrant client](https://github.com/qdrant/qdrant-client) · [embedding fixture model](https://huggingface.co/sentence-transformers-testing/stsb-bert-tiny-safetensors) · [cross-encoder](https://huggingface.co/cross-encoder/ms-marco-TinyBERT-L2-v2)

The optional local generator uses separately downloaded Qwen2.5-0.5B-Instruct weights. [Pinned model and license](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/tree/7ae557604adf67be50417f59c2c2f167def9a775)
