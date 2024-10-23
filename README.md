<div align="center">
<h1><img src="static/images/ShadowKV.png" height="40px"> ShadowKV: KV Cache in Shadows for High-Throughput Long-Context LLM Inference</h1>
</div>
<div align="center">
<b><a href="https://github.com/preminstrel">Hanshi Sun</a></b><sup>1,2</sup>,
<b><a href="https://lchang20.github.io/">Li-Wen Chang</a></b><sup>2</sup>,
<b><a href="https://sites.google.com/view/wenleibao/">Wenlei Bao</a></b><sup>2</sup>,
<b><a href="https://sizezheng.github.io/">Size Zheng</a></b><sup>2</sup>,
<b><a href="https://zheng-ningxin.github.io/">Ningxin Zheng</a></b><sup>2</sup>,
<b><a href="https://scholar.google.com/citations?user=ZMfk2F8AAAAJ&hl=zh-CN">Xin Liu</a></b><sup>2</sup>,
<br>
<b><a href="https://github.com/preminstrel">Harry Dong</a></b><sup>1</sup>,
<b><a href="https://github.com/preminstrel">Yuejie Chi</a></b><sup>1</sup>,
<b><a href="https://github.com/preminstrel">Beidi Chen</a></b><sup>1</sup>
</div>
<div align="center">
<sup>1</sup>Carnegie Mellon University
<sup>2</sup>ByteDance
</div>
<div align="center">
[<a href="XXXX">Paper</a>] | [<a href="XXXX">Blog</a>]
</div>
<br>

<div align="center">
<img src="static/images/framework.png" align="top"/>
<figcaption>ShadowKV Framework</figcaption>
</div>

## Environment Set Up
```bash
# create env
conda create -n ShadowKV python=3.10 -y
conda activate ShadowKV

# install packages
pip install -r requirements.txt
pip install flash-attn --no-build-isolation

# nemo dependencies (for dataset building)
pip install wheel
pip install Cython
pip install youtokentome
pip install nemo_toolkit[all]==1.23

# flashinfer
pip install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.3/
pip install huggingface_hub==0.22.0

# build kernels
python setup.py build_ext --inplace
```
## Supported Models
Currently, we support the following LLMs:
- Llama-3-8B-1M: [gradientai/Llama-3-8B-Instruct-Gradient-1048k](https://huggingface.co/gradientai/Llama-3-8B-Instruct-Gradient-1048k)
- GLM-4-9B-1M: [THUDM/glm-4-9b-chat-1m](https://huggingface.co/THUDM/glm-4-9b-chat-1m)
- Llama-3.1-8B: [meta-llama/Meta-Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct)
- Yi-9B-200K: [01-ai/Yi-9B-200K](https://huggingface.co/01-ai/Yi-9B-200K)
- Phi-3-Mini-128K: [microsoft/Phi-3-mini-128k-instruct](https://huggingface.co/microsoft/Phi-3-mini-128k-instruct)
- Qwen2-7B-128K: [Qwen/Qwen2-7B-Instruct](https://huggingface.co/Qwen/Qwen2-7B-Instruct)

## Accuracy Evaluations
Here we provide an example to build the dataset and run evaluation for the [RULER](https://github.com/hsiehjackson/RULER) benchmark with Llama-3-8B-1M.

### Build Dataset

```bash
# build RULER
python -c "import nltk; nltk.download('punkt')"
cd data/ruler
bash create_dataset.sh "gradientai/Llama-3-8B-Instruct-Gradient-1048k" "llama-3"
```

### Run Evaluation

```bash
# Full attention
OMP_NUM_THREADS=48 torchrun --standalone --nnodes=1 --nproc_per_node 8 test/eval_acc.py --datalen 131072 --method full --dataset_name "ruler/niah_single_1,ruler/niah_single_2,ruler/niah_single_3,ruler/niah_multikey_1,ruler/niah_multikey_2,ruler/niah_multikey_3,ruler/niah_multiquery,ruler/niah_multivalue,ruler/vt,ruler/cwe,ruler/fwe,ruler/qa_1,ruler/qa_2" --model_name "gradientai/Llama-3-8B-Instruct-Gradient-1048k"

# ShadowKV
OMP_NUM_THREADS=48 torchrun --standalone --nnodes=1 --nproc_per_node 8 test/eval_acc.py --datalen 131072 --method shadowkv --dataset_name "ruler/niah_single_1,ruler/niah_single_2,ruler/niah_single_3,ruler/niah_multikey_1,ruler/niah_multikey_2,ruler/niah_multikey_3,ruler/niah_multiquery,ruler/niah_multivalue,ruler/vt,ruler/cwe,ruler/fwe,ruler/qa_1,ruler/qa_2" --sparse_budget 2048 --rank 160 --chunk_size 8
```
