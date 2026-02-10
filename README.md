<h1 align="center">SILVIA
  <br/>
  <sub>Segmentation and Identification for satelLite Vegetation pattern ImAges</sub>
</h1>

<p align="center">
  <a href="https://python.org"><img alt="Python" src="https://img.shields.io/badge/Python-3.9+-3776AB?logo=python&logoColor=white"></a>
  <img alt="OS" src="https://img.shields.io/badge/OS-Linux%20%7C%20macOS%20%7C%20Windows-informational">
  <a href="https://github.com/facebookresearch/segment-anything"><img alt="SAM2" src="https://img.shields.io/badge/SAM-v2-blue"></a>
  <a href="https://github.com/Ste-lla02/silvia/stargazers"><img alt="GitHub stars" src="https://img.shields.io/github/stars/Ste-lla02/silvia?style=social"></a>
</p>

<p align="center">
  <img src="Figures/silvia_pipeline.png" alt="SILVIA Pipeline Workflow" width="80%">
</p>

**SILVIA** is a modular image segmentation pipeline designed for the detection and analysis of vegetation patterns in high-resolution satellite imagery.

The framework integrates the Segment Anything Model (SAM) with domain-specific preprocessing, geometric filtering, and multi-channel mask fusion techniques to support robust and reproducible environmental analysis.

**Relation to Other Projects**  

SILVIA is the segmentation and analysis component of the **TITANIA** framework, a modular architecture designed to integrate satellite data acquisition and vegetation pattern analysis within a unified and reproducible workflow.

In particular, SILVIA is designed to process data acquired through **[DAPHNE](https://github.com/stefanomarrone/daphne)**, the data acquisition module of the TITANIA framework.

Beyond environmental monitoring, the SILVIA pipeline has been adapted to other application domains, including medical image analysis. This cross-domain reuse highlights the domain-independent design of the pipeline and its suitability for different image-based analysis tasks.


---

## Motivation

Vegetation patterns such as fairy circles, gaps, rings, and clustered structures are important indicators of ecosystem dynamics, soil conditions, and environmental stress in arid and semi-arid regions.  
However, automated detection of these patterns from satellite imagery remains challenging due to:

- heterogeneous image sources and resolutions,
- absence of labelled datasets,
- sensitivity to noise (e.g., clouds, shadows),
- variability in vegetation appearance.

SILVIA addresses these challenges by combining **foundation-model-based segmentation** with **pre- and post-processing, and fusion strategies**, without requiring supervised training.

---

## Main Features

- **SAM-based segmentation**  
  Automatic generation of segmentation masks without task-specific training.

- **Multi-representation processing**  
  - RGB satellite images,
  - individual spectral channels,
  - vegetation index–derived images (e.g. ExG, NDVI when available).

- **Configurable geometric filtering**
  - area constraints,
  - roundness,
  - eccentricity,
  - segmentation stability and quality scores.

- **Multi-channel mask fusion**  
  Voting-based mechanisms to:
  - reduce false positives (FP),
  - reduce false negatives (FN),  
  depending on the selected fusion policy.

- **Research-oriented and extensible**  
  Designed to support experimentation, reproducibility, and adaptation to different domains.

---

## Pipeline Overview

The SILVIA pipeline is structured into four main stages:

1. **Preprocessing**
   - image resizing and tiling,
   - spectral decomposition and vegetation index computation,
   - optional enhancement filters.

2. **Mask Generation**
   - application of SAM to each image representation,
   - extraction of geometric and quality attributes.

3. **Filtering**
   - removal of low-quality or geometrically inconsistent masks.

4. **Channel Fusion**
   - voting-based fusion of masks obtained from different channels.


---
## Dependencies

The framework is implemented in Python (version 3.9 recommended).

All required Python packages and external dependencies are listed in the `requirements.txt` file provided in the repository. The recommended way to install the dependencies is:

```bash
pip install -r requirements.txt
```

---
## Usage

SILVIA is designed to be configured through user-defined parameters that control the behaviour of the pipeline, including:

- segmentation model settings,
- preprocessing options,
- filtering thresholds,
- fusion policies.

A typical workflow involves the following steps:

1. selecting a satellite image to analyse,
2. configuring preprocessing and segmentation parameters,
3. running the pipeline to obtain filtered and fused segmentation masks.

Detailed usage examples and configuration files will be progressively added to the repository.

---
## Output

The SILVIA pipeline produces the following outputs:

- filtered segmentation masks,
- fused mask representations obtained through channel voting,
- geometric and quality metadata associated with each detected structure.

These outputs can be used for:

- spatial analysis of vegetation patterns,
- temporal comparison across different acquisition times,
- environmental monitoring and ecosystem studies.

---

## Replication Package

A replication package will be provided to support reproducibility of the experiments reported in the scientific publications associated with SILVIA.

The package will be released under the `replication/` directory and will include:
- configuration files used to run the pipeline on the case studies,
- scripts to reproduce the main results,
- example inputs (or instructions to retrieve them, when redistribution is not possible),
- output artifacts (e.g., masks, logs, and computed metrics) required to replicate the reported experiments.

At the current stage, this material is **under preparation** and will be added in a future update of the repository.

---

## License
The software is licensed according to the GNU General Public License v3.0 (see License file).

---
## Citation and Contact

**Citation**  

If you find this work useful for your research, please consider citing:

```bibtex
@article{silvia2025,
  title = {Detecting Vegetation Patterns in Satellite Images: {SILVIA}, a Segmentation-Based Approach},
  author = {de Biase, Maria Stella and De Fazio, Roberta and Marrone, Stefano},
  journal = {Procedia Computer Science},
  year = {2025},
  doi = {10.1016/j.procs.2025.10.020}
}
```

**Contact**  
For questions or collaborations:


**Maria Stella de Biase**  
University of Campania “Luigi Vanvitelli”  
📧 mariastella.debiase@unicampania.it

**Stefano Marrone**  
University of Campania “Luigi Vanvitelli”  
📧 stefano.marrone@unicampania.it

---
