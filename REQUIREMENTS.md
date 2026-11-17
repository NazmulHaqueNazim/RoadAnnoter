# RoadAnnoter requirements

Everything needed to run RoadAnnoter on a new machine: system requirements,
Python packages, model checkpoints with download links, and the licence terms
of each model.

Last updated: 17 November 2026.

## 1. System requirements

| Item | Requirement |
| --- | --- |
| Operating system | Windows 10/11 (the only system the tool has been used on). Linux and macOS should work but are untested. |
| Python | 3.11 or newer. Developed on Python 3.14. Avoid 3.14.1: the current torchvision release excludes that exact version. |
| RAM | 8 GB minimum, 16 GB recommended for SAM ViT-L/ViT-H or SegFormer-B5. |
| GPU | Optional. An NVIDIA GPU with CUDA makes SAM, SegFormer and training much faster. Everything runs on CPU. |
| Disk | About 2 GB for packages, plus the checkpoints you choose (40 MB to 2.6 GB each). |
| Internet | Needed for installation and for the first use of each SegFormer or torchvision model. Not needed afterwards. |

## 2. Python packages

| Package (pip name) | Imported as | Minimum | Needed for | Licence |
| --- | --- | --- | --- | --- |
| PyQt5 | `PyQt5` | 5.15 | The whole application | GPL v3 |
| opencv-python | `cv2` | 4.8 | The whole application, Edge Fill | Apache 2.0 |
| numpy | `numpy` | 1.24 | The whole application | BSD-3-Clause |
| torch | `torch` | 2.0 | SAM, training, all model suggestions | See the PyTorch licence |
| torchvision | `torchvision` | 0.15 | Training, FCN/DeepLabV3 suggestions | BSD-3-Clause |
| transformers | `transformers` | 4.30 | SegFormer suggestions | Apache 2.0 |
| pillow | `PIL` | 9.0 | SegFormer suggestions | MIT-CMU |
| segment-anything | `segment_anything` | 1.0 | Original SAM checkpoints | Apache 2.0 |
| mobile_sam (from GitHub) | `mobile_sam` | 1.0 | MobileSAM checkpoint | Apache 2.0 |
| timm | `timm` | 0.9 | Required by `mobile_sam` | Apache 2.0 |

Notes:

- The minimum versions are conservative floors, not tested limits. The only
  tested configuration is the developer's own machine (Python 3.14,
  OpenCV 5.0.0). Record that exact set with `pip freeze > requirements-lock.txt`
  and commit the file, so others can reproduce it.
- `mobile_sam` is not on PyPI and must be installed from GitHub. It is required
  for `mobile_sam.pt`, not optional.
- `mobile_sam` prints a few `timm` deprecation warnings at start-up. They are
  harmless and RoadAnnoter hides them.
- Without the `mobile_sam` package, `mobile_sam.pt` still loads, through a
  built-in substitute encoder, but the masks are much worse (see
  Troubleshooting). After loading, the SAM panel must show `[mobile_sam]`.
- The code also tries to import `ultralytics`. It is not used by any feature
  and does not need to be installed.

### pip install commands

Each package on its own line, in the order to run them:

```bash
# Core: the application, manual polygons, Edge Fill
pip install "PyQt5>=5.15"
pip install "opencv-python>=4.8"
pip install "numpy>=1.24"

# PyTorch: SAM, training, all model suggestions
# (GPU users: use the command from https://pytorch.org/get-started/locally/ instead)
pip install "torch>=2.0" "torchvision>=0.15"

# SegFormer (Cityscapes) suggestions
pip install "transformers>=4.30"
pip install "pillow>=9.0"

# Original SAM checkpoints (ViT-B / ViT-L / ViT-H)
pip install "segment-anything>=1.0"

# MobileSAM checkpoint (not on PyPI; installed from GitHub, no Git needed)
pip install "timm>=0.9"
pip install https://github.com/ChaoningZhang/MobileSAM/archive/refs/heads/master.zip
```

Everything in one go:

```bash
pip install "PyQt5>=5.15" "opencv-python>=4.8" "numpy>=1.24" "torch>=2.0" "torchvision>=0.15" "transformers>=4.30" "pillow>=9.0" "segment-anything>=1.0" "timm>=0.9"
pip install https://github.com/ChaoningZhang/MobileSAM/archive/refs/heads/master.zip
```

Or use the requirements file, which installs the same set:

```bash
pip install -r requirements.txt
```

By feature, if you do not want everything:

| You want | Command |
| --- | --- |
| Manual polygons and Edge Fill only | `pip install PyQt5 opencv-python numpy` |
| + MobileSAM (SAM prompt, Smart Select) | `pip install torch torchvision timm` then `pip install https://github.com/ChaoningZhang/MobileSAM/archive/refs/heads/master.zip` |
| + original SAM | `pip install torch torchvision segment-anything` |
| + SegFormer suggestions | `pip install torch transformers pillow` |
| + FCN / DeepLabV3 suggestions, training | `pip install torch torchvision` |

Keep the quotes around names with `>=`; without them the shell treats `>` as a
redirect.

### What each feature needs

| Feature | Packages | Checkpoint |
| --- | --- | --- |
| Manual polygons, editing, Edge Fill (W) | PyQt5, opencv-python, numpy | None |
| SAM prompt (S), Smart Select (Q) with MobileSAM | + torch, torchvision, mobile_sam, timm | `mobile_sam.pt` |
| SAM prompt (S), Smart Select (Q) with original SAM | + torch, torchvision, segment-anything | `sam_vit_b/l/h` |
| SegFormer suggestions | + torch, transformers, pillow | Downloaded automatically |
| FCN / DeepLabV3 suggestions | + torch, torchvision | Downloaded automatically |
| Training and own-model suggestions | + torch, torchvision | MobileNetV3-Small backbone, downloaded automatically |

## 3. Installation

```bash
# 1. Create and activate a virtual environment (recommended)
python -m venv .venv
.venv\Scripts\activate            # Windows
# source .venv/bin/activate       # Linux / macOS

# 2. GPU only: install torch and torchvision first, using the command that
#    https://pytorch.org/get-started/locally/ gives for your CUDA version.
#    CPU users can skip this step.

# 3. Install everything else
pip install -r requirements.txt

# 4. Run
python seg_annotator.py
```

The MobileSAM line installs from a GitHub zip archive, so Git is not needed.
If you have Git and prefer it, `pip install git+https://github.com/ChaoningZhang/MobileSAM.git`
installs the same package.

Step 3 can be replaced by any of the commands under "pip install commands" in
section 2.

Check the installation:

```bash
python -c "import PyQt5, cv2, numpy, torch, torchvision, transformers, PIL, segment_anything, mobile_sam, timm; print('all imports OK; CUDA available:', torch.cuda.is_available())"
```

## 4. Checkpoints you download yourself (SAM)

Download one of these, save it anywhere, and load it in the app with
**Annotate > Load SAM checkpoint**. The model type is detected automatically.

The pip packages contain only the model code. The checkpoint file holds the
trained weights and is always needed as well; installing the package does not
download it.

| Model | File | Approx. size | Download link |
| --- | --- | --- | --- |
| MobileSAM (recommended: fast, small) | `mobile_sam.pt` | 40.7 MB | https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt |
| SAM ViT-B | `sam_vit_b_01ec64.pth` | about 375 MB | https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth |
| SAM ViT-L | `sam_vit_l_0b3195.pth` | about 1.2 GB | https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth |
| SAM ViT-H (most accurate, slowest) | `sam_vit_h_4b8939.pth` | about 2.6 GB | https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth |

Project pages: https://github.com/ChaoningZhang/MobileSAM and
https://github.com/facebookresearch/segment-anything

## 5. Weights downloaded automatically

These download on first use and are then read from the local cache. The links
are given so the files can be fetched by hand on a slow or offline machine.

### SegFormer, fine-tuned on Cityscapes (recommended for road scenes)

Downloaded by the `transformers` library from Hugging Face. Sizes are estimates
from the published parameter counts.

| Model in the app | Hugging Face model id | Approx. size | Model page |
| --- | --- | --- | --- |
| SegFormer-B0 (fastest) | `nvidia/segformer-b0-finetuned-cityscapes-512-1024` | about 15 MB | https://huggingface.co/nvidia/segformer-b0-finetuned-cityscapes-512-1024 |
| SegFormer-B2 (balanced) | `nvidia/segformer-b2-finetuned-cityscapes-1024-1024` | about 110 MB | https://huggingface.co/nvidia/segformer-b2-finetuned-cityscapes-1024-1024 |
| SegFormer-B5 (best quality) | `nvidia/segformer-b5-finetuned-cityscapes-1024-1024` | about 340 MB | https://huggingface.co/nvidia/segformer-b5-finetuned-cityscapes-1024-1024 |

The size labels shown in the app's model list are lower than these for B2 and
B5; treat the figures here as the better estimate.

Cache folder: `C:\Users\<name>\.cache\huggingface\hub` on Windows,
`~/.cache/huggingface/hub` on Linux and macOS.

To pre-download a model (for example before going offline):

```bash
python -c "from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor as P; m='nvidia/segformer-b0-finetuned-cityscapes-512-1024'; P.from_pretrained(m); SegformerForSemanticSegmentation.from_pretrained(m)"
```

### torchvision models

Downloaded by `torchvision` from the PyTorch site. The three segmentation
models were trained on a subset of COCO using the 20 Pascal VOC categories, so
they detect vehicles and people but have no road class.

| Used for | File | Size | Download link |
| --- | --- | --- | --- |
| FCN ResNet-101 suggestions | `fcn_resnet101_coco-7ecb50ca.pth` | 208 MB | https://download.pytorch.org/models/fcn_resnet101_coco-7ecb50ca.pth |
| DeepLabV3 ResNet-101 suggestions | `deeplabv3_resnet101_coco-586e9e4e.pth` | 233 MB | https://download.pytorch.org/models/deeplabv3_resnet101_coco-586e9e4e.pth |
| DeepLabV3 MobileNetV3-Large suggestions | `deeplabv3_mobilenet_v3_large-fc3c493d.pth` | 42 MB | https://download.pytorch.org/models/deeplabv3_mobilenet_v3_large-fc3c493d.pth |
| Backbone for training your own model | `mobilenet_v3_small-047dcff4.pth` | 10 MB | https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth |

Cache folder: `C:\Users\<name>\.cache\torch\hub\checkpoints` on Windows,
`~/.cache/torch/hub/checkpoints` on Linux and macOS.

If a download is slow, fetch the file in a browser and place it in that folder
with its name unchanged. The app then finds it and skips the download.

## 6. Moving to another machine without re-downloading

1. Copy the repository folder.
2. Copy your SAM checkpoint file (`mobile_sam.pt` or `sam_vit_*.pth`). It is
   not in the cache folders.
3. Copy the two cache folders named in section 5 to the same place under the
   new user's home folder (`C:\Users\<new name>\.cache\...`). They contain only
   the models that were actually used on the old machine; any other model
   downloads on first use.
4. Install the packages (section 3). Copying the cache does not install them.
   Use `requirements-lock.txt` if you created one, to get identical versions.
5. Check: load the checkpoint and confirm the SAM panel shows `[mobile_sam]`;
   then generate SegFormer suggestions on one image and confirm no download
   starts.

## 7. Troubleshooting

| Symptom | Cause | Fix |
| --- | --- | --- |
| SAM masks are small or fragmented; the SAM panel shows `[built-in bridge + segment_anything]`; the model-type list has only `[segment_anything]` entries | The `mobile_sam` package is not installed in the Python that runs the app | `pip install timm` then `pip install https://github.com/ChaoningZhang/MobileSAM/archive/refs/heads/master.zip`, restart the app, reload the checkpoint |
| Label still shows `built-in bridge` after installing | The package went into a different Python | Run `python -c "import mobile_sam, sys; print(sys.executable)"` and install with that same interpreter, e.g. `py -3.14 -m pip install ...` |
| `timm` FutureWarning or "Overwriting tiny_vit_..." messages | Normal start-up notices from `mobile_sam` | None needed |
| "SAM: not loaded" | No checkpoint chosen yet | Load `mobile_sam.pt` (section 4) |
| A model download is very slow | Throttled connection | Download the file in a browser and place it in the cache folder (section 5) |
| SegFormer option fails with "Transformers not installed" | Missing package | `pip install transformers pillow` |

## 8. Licence terms of the models

RoadAnnoter does not redistribute any model weights; each user downloads them
from the original source and must follow that source's terms.

| Model | Terms |
| --- | --- |
| SAM (code and checkpoints) | Apache 2.0 |
| MobileSAM (code and checkpoint) | Apache 2.0 |
| SegFormer checkpoints | NVIDIA Source Code License for SegFormer. Read it before any use outside research: https://github.com/NVlabs/SegFormer/blob/master/LICENSE |
| Cityscapes (data the SegFormer checkpoints were trained on) | Cityscapes dataset terms: https://www.cityscapes-dataset.com/license/ |
| torchvision code | BSD-3-Clause. The weights were trained on COCO; see https://cocodataset.org/#termsofuse |
| PyQt5 | GPL v3, which governs how an application built on it may be distributed |
