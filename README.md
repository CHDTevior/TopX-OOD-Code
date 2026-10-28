# TopX: Text-to-Motion for Unseen Skeletons from the Skeleton File Alone

Tengjiao Sun, Pengcheng Fang, Xiaoyu Zhan, Dongjie Fu, Xiaohao Cai, Hansung Kim

**Paper** (arXiv, link coming soon) · **Weights** ([HuggingFace](https://huggingface.co/Tevior/TopX-OOD)) · **Project page** (coming soon)

TopX is a 73M-parameter flow-matching transformer that generates joint rotations and a root trajectory for any
skeleton from its kinematic tree, rest pose and joint names, without clips of the character, templates, retargeting or
fine-tuning. Motion is normalised by the rest pose and the size of the skeleton, and the rest pose is encoded as a
world-frame descriptor. Attention is factorised over joints and frames; across joints, queries and keys are rotated by
**HK-RoPE**, a heat-kernel rotary encoding of the skeleton graph that is defined for every tree and does not change
with the order of its joints. A single model animates skeletons from 4 to 117 joints and generates up to 240 frames.

This repository contains the model and the generation script for a new skeleton. Training code and the SkelShift
benchmark (data rebuild scripts and evaluation) will be added.

## Installation

```bash
git clone https://github.com/CHDTevior/TopX-OOD-Code.git
cd TopX-OOD-Code
pip install -r requirements.txt
```

Download the weights into `weights/`:

```bash
huggingface-cli download Tevior/TopX-OOD topx.pt --local-dir weights
```

## Demos

`demo/` holds nine skeletons with stored caption and joint-description embeddings, so the demos run without the text
encoder:

```bash
bash demo/run_demos.sh            # outputs in out/demos/
```

See `demo/README.md` for the prompts and the sources of the skeletons.

## Your own skeleton

```bash
python scripts/generate.py --ckpt weights/topx.pt --skeleton my_rig.bvh --up +Y --forward +Z \
    --text "An object walks forward and sits down." --frames 120 --out out/
```

* `--skeleton` takes a BVH file (the rest pose is frame `--rest_frame`, default 0) or a skeleton `.npz` in the format of
  `demo/*/skeleton.npz`. For a BVH file, `--up` and `--forward` name the axes of the file that point up and that the
  character faces (write negative axes as `--forward=-Z`); a skeleton `.npz` must already be +Y up and +Z
  forward.
* Run with `--describe_only` first to check the rest pose and the joint descriptions the model will read.
* `--text` encodes the prompt with LLM2Vec (`McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp` with its supervised
  adapter, built on the gated `meta-llama/Meta-Llama-3-8B-Instruct`; about 16 GB in bf16). The training captions read
  "An object &lt;does something&gt;."; on a new character, also try naming it. `--text_emb` and `--joint_sem` take
  precomputed embeddings instead.
* Generation uses 50 Euler steps and classifier-free guidance 2 by default (`--steps`, `--cfg_text`), with TF32 matrix
  products on CUDA (`--tf32 0` turns them off).

Each run writes `<name>.npz` (joint positions from the position channels and from forward kinematics of the rotations,
the rest-relative rotations and the model output), `<name>.gif` (rest pose and both decodes) and `<name>.bvh` (the input
hierarchy animated with the generated rotations and root translation, in the input's axes and units).

## Code layout

| path | content |
|---|---|
| `scripts/generate.py` | text-to-motion on one skeleton: reading BVH / npz rigs, joint descriptions, sampling, BVH export |
| `src/models/v2/dit_motion.py` | the factorised joint-by-frame transformer and the flow-matching sampler |
| `src/models/v2/hk_rope.py` | HK-RoPE, the heat-kernel rotary encoding of the joint axis |
| `src/models/v2/frame_rope.py` | the rotary encoding of the frame axis |
| `src/data/skeleton_spectral.py` | the heat-kernel signature of a skeleton tree |
| `src/data/rig_features.py` | per-joint structural and world-frame rest features, batch collation |
| `src/data/ktjd17/` | the per-joint motion representation: encoding and decoding |
| `src/data/joint_descriptions.py` | joint descriptions from the skeleton's geometry |

## Acknowledgements

The AdaLN conditioning of the transformer blocks and the output layer follow DiT (Peebles and Xie, 2023), and
the sinusoidal timestep embedding follows [guided-diffusion](https://github.com/openai/guided-diffusion). Text is
encoded with [LLM2Vec](https://github.com/McGill-NLP/llm2vec).

## License

The code is released under the MIT License (`LICENSE`). The model weights are released under
[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). The demo skeletons keep the licences of their creators
and the annotations theirs; see `demo/README.md`.

## Citation

```bibtex
@article{sun2026topx,
  title   = {TopX: Text-to-Motion for Unseen Skeletons from the Skeleton File Alone},
  author  = {Sun, Tengjiao and Fang, Pengcheng and Zhan, Xiaoyu and Fu, Dongjie and Cai, Xiaohao and Kim, Hansung},
  journal = {arXiv preprint},
  year    = {2026}
}
```
