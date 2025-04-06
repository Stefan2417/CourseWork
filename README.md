# The Impact of Language on Automatic Speaker Verification Systems

Course project at HSE (2025), supervised by Petr Grinberg. It studies how multilingual self-supervised pretraining affects speaker verification (SV) when training and test languages differ and when data for the target language is scarce.

Full report: [CourseWork.pdf](./CourseWork.pdf). Checkpoints: [Hugging Face](https://huggingface.co/Sttefan/speaker-verification-checkpoints).

## Approach

Two multilingual SSL models are adapted to speaker verification with PMFA: hidden states from selected layers are concatenated, aggregated with attentive statistics pooling and trained with an AAM-Softmax head.

- [Wav2Vec2-BERT 2.0](https://huggingface.co/facebook/w2v-bert-2.0)
- [XEUS](https://huggingface.co/espnet/xeus) (multilingual, E-Branchformer)
- ECAPA-TDNN as a supervised baseline

All models are trained on VoxCeleb2. They are evaluated on VoxCeleb1-O, VoxSRC21 validation, SL-Celeb (Tamil, Sinhala) and an isiZulu trial list built for this project from the NCHLT corpus, balanced by label and gender.

## Results

Equal Error Rate, lower is better.

| Model            | VoxCeleb1-O | SL-Celeb Tamil | isiZulu | VoxSRC21 val |
|------------------|-------------|----------------|---------|--------------|
| ECAPA-TDNN       | 1.42%       | 3.27%          | 3.30%   | 5.05%        |
| XEUS + PMFA      | 1.29%       | 6.01%          | 2.50%   | 4.06%        |
| W2V2-BERT + PMFA | **0.46%**   | **1.39%**      | **1.90%** | **1.82%**  |

Wav2Vec2-BERT with PMFA gives the lowest EER on every test set. XEUS beats the baseline on three of four sets but is worse on Tamil.

## Running the code

```bash
git clone https://github.com/Stefan2417/CourseWork.git
cd CourseWork
git lfs install
git clone https://huggingface.co/espnet/XEUS
pip install -r requirements.txt
./scripts/download_test_pairs_voxceleb.sh
```

Data:

- [VoxCeleb1 and VoxCeleb2](https://huggingface.co/datasets/ProgramComputer/voxceleb) for training and evaluation
- [NCHLT isiZulu Speech Corpus](https://repo.sadilar.org/handle/20.500.12185/275) for evaluation
- [SLCeleb](https://ieee-dataport.org/documents/slceleb-speaker-verification) for evaluation (access on request)
- [MUSAN](https://www.openslr.org/17) and [RIRS_NOISES](https://www.openslr.org/28) for augmentation

Configs live in `src/configs`. Set the absolute paths to datasets, trial lists, checkpoints and output directories there, then run:

```bash
python train.py --config src/configs/<config>.yaml
python inference.py --config src/configs/<config>.yaml
```

Weights & Biases logging reads the key from the `WANDB_API_KEY` environment variable.

## Acknowledgements

Experiments were run on the HSE supercomputing cluster. The project structure is based on [Blinorot/pytorch_project_template](https://github.com/Blinorot/pytorch_project_template).
