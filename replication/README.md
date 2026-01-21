# Replication Package

This directory provides a minimal and self-contained replication package
to reproduce an example run of the segmentation and fusion
pipeline described in the paper.

The goal of this package is to demonstrate how the proposed methodology
can be executed end-to-end on a sample input image using the provided
configuration file.

## Notes on reproducibility

This replication package is designed to reproduce a single example run
of the proposed methodology.

Due to the stochastic nature of the segmentation model, minor variations
in the generated masks may occur. However, the overall pipeline behavior
and the structure of the produced outputs are deterministic.

The provided input image (in `replication/data/src` directory) and configuration (in `replication/config` directory) ensure that the example can
be executed without requiring additional datasets or credentials.

## Requirements

The replication package was tested with:

- Python >= 3.9
- Linux / macOS

All required Python dependencies are listed in `requirements.txt`.

To install the dependencies, run:

```bash
pip install -r requirements.txt
```

## How to run the example

From the `replication/` directory, run:

```bash
python run_example.py
```

## Expected output

After successful execution, the following outputs are generated:

- cropped images in the `replication/data/cropped` directory
- segmented masks stored in the `replication/data/mask` directory
- fused masks produced by the voting strategy in the `replication/data/fusion` directory



## Citation

If you use this software or the replication package, please cite the
associated paper.