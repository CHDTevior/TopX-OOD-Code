# Demo skeletons

Ready-to-run examples for `scripts/generate.py`: nine skeletons, each with the caption and joint-description
embeddings stored, so no text encoder is needed. Each folder holds `skeleton.npz`, `joint_sem.npy`,
`text_emb_<k>.npy`, `prompts.json` and the reference output `ref_<k>.gif` / `ref_<k>.bvh` of each prompt, made with
`bash demo/run_demos.sh` on an NVIDIA H200 (another GPU may differ in the last digits).

```bash
bash demo/run_demos.sh            # CKPT=weights/topx.pt, outputs in out/demos/
```

| demo | joints | prompt | seed | frames |
|---|---|---|---|---|
| `bat` | 48 | An object flaps its wings in place. | 7 | 60 |
| `bear` | 20 | An object walks in place. | 17 | 200 |
| `crawling_human` | 44 | An object swims in place with arms and legs moving. | 17 | 149 |
| `mech_striker` | 16 | An object walks forward. | 17 | 31 |
| `mech_striker` | 16 | An object kicks with one leg. | 17 | 26 |
| `spider` | 24 | An object scuttles in place with legs moving. | 17 | 31 |
| `stag` | 28 | An object rears up and lowers its head. | 7 | 92 |
| `triceratops` | 29 | An object walks in place. | 7 | 61 |
| `triceratops` | 29 | An object stomps its feet in place. | 7 | 130 |
| `viking_worker` | 30 | An object falls backward and lies on its back. | 17 | 68 |
| `viking_worker` | 30 | An object crouches and falls forward. | 17 | 34 |
| `wolf_lowpoly` | 37 | An object crouches and then stands while flapping its arms. | 7 | 113 |

## Sources and licences

The skeletons come from 3D models published on Sketchfab and indexed by Objaverse-XL, taken from the UniML3D export of
Objaverse-XL. Each model keeps its creator's licence, listed below as shown on Sketchfab on the retrieval date in its
`prompts.json`; nothing here grants rights beyond it. Only the skeleton (joint hierarchy, rest pose and joint names) was
extracted from each model; no mesh, texture or animation is included.

The prompts are UniML3D captions and the joint descriptions are built from UniML3D's cleaned joint labels. UniML3D
(https://huggingface.co/datasets/Linzhan/UniML3D) offers these annotations under
[ODC-BY 1.0](https://opendatacommons.org/licenses/by/1-0/); please cite

```bibtex
@article{mou2026unimate,
  title   = {UniMate: One Unified Model to Animate Diverse Skeletons},
  author  = {Mou, Linzhan and Lei, Jiahui and Dou, Zhiyang and Cai, Chenyue and Song, Chaoyue and Finkelstein, Adam and Rusinkiewicz, Szymon},
  journal = {arXiv preprint arXiv:2609.05415},
  year    = {2026}
}
```

If you are a rights holder and want a skeleton removed, open an issue on this repository.

| demo | model | author | licence | source |
|---|---|---|---|---|
| `bat` | Bat Bay04 | thanhnhan | [CC BY 4.0](http://creativecommons.org/licenses/by/4.0/) | https://sketchfab.com/3d-models/bat-bay04-614ab66892894acabcda5cc4a94f87fe |
| `bear` | Bear Walk | tiikeri | [CC BY 4.0](http://creativecommons.org/licenses/by/4.0/) | https://sketchfab.com/3d-models/bear-walk-ffd55e32c04c498681ed11584bdd49a5 |
| `crawling_human` | Crawling mutated human | Elisey | [CC BY 4.0](http://creativecommons.org/licenses/by/4.0/) | https://sketchfab.com/3d-models/crawling-mutated-human-a87532a3e89947159cc1303008c06eaf |
| `mech_striker` | Medium Mech Striker | MSGDI | [CC BY 4.0](http://creativecommons.org/licenses/by/4.0/) | https://sketchfab.com/3d-models/medium-mech-striker-27ba717c173a40b7841d2f2c6a89d823 |
| `spider` | Blackarachnia | PCIXOPAT | [CC BY 4.0](http://creativecommons.org/licenses/by/4.0/) | https://sketchfab.com/3d-models/blackarachnia-237ce5d21a3c4bcab51f34a4ab451587 |
| `stag` | Roaring Stag ( deepdreamed ) | Miguelangelo Rosario | [CC BY 4.0](http://creativecommons.org/licenses/by/4.0/) | https://sketchfab.com/3d-models/roaring-stag-deepdreamed-4798d8c87a0e4ad8835217fe93ddf67b |
| `triceratops` | Triceratops occultatum | Miguelangelo Rosario | [CC BY 4.0](http://creativecommons.org/licenses/by/4.0/) | https://sketchfab.com/3d-models/triceratops-occultatum-d8b6a381f36c46f8b1d59ed6e0b57c65 |
| `viking_worker` | Viking Framps Worker | RG3D | [CC BY 4.0](http://creativecommons.org/licenses/by/4.0/) | https://sketchfab.com/3d-models/viking-framps-worker-cef34035d74c4dfdb9cad45fa36da294 |
| `wolf_lowpoly` | Low poly Snake - Wolf Animated | imitate | [CC BY 4.0](http://creativecommons.org/licenses/by/4.0/) | https://sketchfab.com/3d-models/low-poly-snake-wolf-animated-c0baad2baee5467894087856cac1872b |
