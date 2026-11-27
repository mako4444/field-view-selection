# field-view-selection

Code for the experiments of Chapter 4, *Using the Field to Select Viewpoints for Localization*, of the
PhD thesis *Image Detectability as 3D Saliency for Visual Localization* (Han Sun). The chapter is based on
the paper https://arxiv.org/abs/2611.34139.

The experiments evaluate viewpoint selection with an offline replay protocol. Every candidate view of a
decision is localized once with a fixed 2D-3D localizer (SuperPoint, EPnP-RANSAC); a planner sees only the
map and the candidate poses, and its top-1 success is the localization outcome of the candidate it picks.
DetField, the detectability field of Chapter 3, is used frozen.

## Setup

```bash
conda env create -f environment.yml && conda activate fvs
pip install "git+https://github.com/facebookresearch/pytorch3d.git@v0.7.9"
pip install -e .
```

A CUDA GPU is required. The results were produced on an RTX 5090 with the environment above. On other
hardware, the dense object experiment and the HM3D transfer can differ slightly, because RANSAC and GPU
sampling are not bit-identical across machines.

## Data

The data is on Zenodo, https://doi.org/10.5281/zenodo.24573550. From the repository root,

```bash
bash scripts/download_data.sh
```

downloads the archives (5.9 GB), checks them and unpacks them into `data/`:

```
data/weights/       DetField weights (objects, dense scenes, sparse scenes)
data/planners/      trained per-landmark planners and LWL models
data/objects/       object point sets and lists
data/indoor/        test and development crops of SceneFun3D, with lists
data/hm3d/          camera trajectories of the HM3D room videos and the crop list
```

External data, under `data/external/` or at the paths given by the environment variables:

| Data | Location | Needed for |
|---|---|---|
| SuperPoint weights `superpoint_v1.pth` ([MagicLeap](https://github.com/magicleap/SuperPointPretrainedNetwork/blob/master/superpoint_v1.pth)) | `data/weights/` | all |
| ShapeNetCore.v2 | `SHAPENET_ROOT` | object experiment |
| SceneFun3D, the videos in `data/indoor/test_videos.txt` and `data/indoor/dev_videos.txt` (`hires_wide`, `hires_depth`, `hires_wide_intrinsics`, `hires_poses.traj`) | `SCENEFUN3D_ROOT` | indoor experiment |
| HM3D scenes `00017-oEPjPNSPmzL` and `00270-bDTsgcSK5Qr` as distributed with ActLoc (Li et al., 2025) | `HM3D_ACTLOC_ROOT` | HM3D transfer |

```bash
wget -P data/weights https://github.com/magicleap/SuperPointPretrainedNetwork/raw/master/superpoint_v1.pth
```

## Experiments

Outputs go to `outputs/` (`FVS_OUT`). Each script runs one experiment end to end.

| Thesis | Command |
|---|---|
| Section 4.5, object experiment (sparse and dense objects) | `bash scripts/run_objects.sh` |
| Section 4.6, indoor experiment (sparse and dense tiers, published baselines) | `bash scripts/run_indoor.sh` |
| Section 4.7, why the gain is small | `bash scripts/run_diagnosis.sh` |
| Section 4.8, transfer to HM3D | `bash scripts/run_hm3d.sh` |
| Tables of Sections 4.5 and 4.6 | `python scripts/make_tables.py --results outputs/results --out outputs/tables` |
| Figure of the paired differences (Section 4.6) | `python scripts/fig_forest.py --results outputs/results --out outputs/figures/forest.pdf` |
| Figures of two indoor test decisions (Sections 4.1 and 4.6) | `bash scripts/fig_indoor.sh` |
| Figure of one object decision (Section 4.5) | `python scripts/fig_obj_setting.py --pick 14,7,0.04,10 --K 16` |

Results are written as JSON files to `outputs/results/`; the table script turns them into the LaTeX tables.

The replay scripts take `--shard i/n`; the number of shards is part of the protocol, because the decisions
of a crop are drawn with the seed of its position within the shard. Shards can run in parallel.

The planners in `data/planners/` are shipped trained; they were trained on separate training rooms of
SceneFun3D, whose data is not part of the data release. The training code is in `fvs.planners.hgb` and
`fvs.planners.lwl`.

DetField values of the indoor crops are stored with the crops. Values computed again on a GPU differ in
the last digits, because the scatter operations of the encoder are not deterministic. The object
experiment computes the field during the replay.

## License

See `LICENSE`.
