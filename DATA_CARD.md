---
dataset_info:
- config_name: osm
  features:
  - name: dsl_building
    dtype: string
  - name: dsl_structure
    dtype: string
  - name: dsl_mass
    dtype: string
  splits:
  - name: train
    num_bytes: 3256969
    num_examples: 12005
  - name: test
    num_bytes: 570318
    num_examples: 2120
  download_size: 1184993
  dataset_size: 3827287
- config_name: synthetic
  features:
  - name: dsl_building
    dtype: string
  - name: dsl_mass
    dtype: string
  - name: dsl_structure
    dtype: string
  - name: dsl_spaces
    dtype: string
  splits:
  - name: train
    num_bytes: 3118607051
    num_examples: 3733546
  - name: valid
    num_bytes: 27495527
    num_examples: 33473
  - name: test
    num_bytes: 110351693
    num_examples: 132928
  download_size: 1149170837
  dataset_size: 3256454271
- config_name: synthetic_rotation_augmented
  features:
  - name: dsl_building
    dtype: string
  - name: dsl_mass
    dtype: string
  - name: dsl_structure
    dtype: string
  - name: dsl_spaces
    dtype: string
  splits:
  - name: train
    num_bytes: 66937105855
    num_examples: 78289138
  - name: valid
    num_bytes: 590262354
    num_examples: 701968
  - name: test
    num_bytes: 2368979456
    num_examples: 2787437
  download_size: 26414017191
  dataset_size: 69896347665
configs:
- config_name: osm
  data_files:
  - split: train
    path: osm/train-*
  - split: test
    path: osm/test-*
- config_name: synthetic
  data_files:
  - split: train
    path: synthetic/train-*
  - split: valid
    path: synthetic/valid-*
  - split: test
    path: synthetic/test-*
- config_name: synthetic_rotation_augmented
  data_files:
  - split: train
    path: synthetic_rotation_augmented/train-*
  - split: valid
    path: synthetic_rotation_augmented/valid-*
  - split: test
    path: synthetic_rotation_augmented/test-*
license: odbl
---

# FLOORA Dataset

Dataset used to train and evaluate **FLOORA**, a domain-specific language model for multifamily residential floor-plan generation.

## Dataset Contents

The dataset provides three configurations.

- **`synthetic`** contains procedurally generated building massings and floor-plan layouts represented in the FLOORA DSL.
- **`synthetic_rotation_augmented`** contains rotation-augmented versions of the synthetic layouts used for model training.
- **`osm`** contains real-world multifamily building footprints derived from OpenStreetMap. These samples contain building massings and metadata but no ground-truth space layouts and are primarily intended for out-of-distribution evaluation.

The DSL represents building metadata, structural information, massing geometry, and, where available, labeled polygons for cores, corridors, and living units.

## License

Open Data Commons Open Database License (ODbL) v1.0.