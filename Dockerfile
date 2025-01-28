FROM python:3.12.4

WORKDIR /workspace

RUN pip install \
    llama-index==0.11.2 \
    huggingface_hub[hf_transfer]==0.24.6 \
    sentence-transformers==3.0.1 \
    torch==2.4.0 \
    chromadb==0.5.5 \
    bs4==0.0.2

CMD ["/bin/bash"]