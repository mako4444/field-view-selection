"""Locations of data, external datasets and outputs. Override with environment variables."""
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get('FVS_DATA', REPO / 'data'))
OUT = Path(os.environ.get('FVS_OUT', REPO / 'outputs'))
SCENEFUN3D = Path(os.environ.get('SCENEFUN3D_ROOT', DATA / 'external/scenefun3d'))
SHAPENET = Path(os.environ.get('SHAPENET_ROOT', DATA / 'external/ShapeNetCore.v2'))
HM3D_ACTLOC = Path(os.environ.get('HM3D_ACTLOC_ROOT', DATA / 'external/hm3d_actloc/full_train_scenes_data'))

WEIGHTS = DATA / 'weights'
SUPERPOINT = WEIGHTS / 'superpoint_v1.pth'
DETFIELD_OBJECTS = WEIGHTS / 'detfield_objects.pt'
DETFIELD_DENSE = WEIGHTS / 'detfield_dense_scenes.pt'
DETFIELD_SPARSE = WEIGHTS / 'detfield_sparse_scenes.pt'
PLANNERS = DATA / 'planners'
OBJECTS = DATA / 'objects'
INDOOR = DATA / 'indoor'
HM3D = DATA / 'hm3d'
