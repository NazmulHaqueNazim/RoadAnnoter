# RoadAnnoter

**Desktop annotation tool for semantic segmentation of road-scene images.**
Combines manual polygon drawing with five assisted methods — SAM/MobileSAM,
Smart Select, Edge Fill, pretrained SegFormer (Cityscapes), and your own
trained model. Saves labels in YOLO segmentation format.

> Developed by **Nazmul Haque**

---

## ✨ Features

| Feature | Key | What it does |
|---|---|---|
| **Manual polygon** | P | Click vertices, close with Enter. Edit mode (E) moves vertices and whole polygons. |
| **SAM prompt** | S | Click foreground/background points; MobileSAM or SAM predicts the mask after each click. |
| **Smart Select** | Q | One click on an object — SAM segments it instantly. Preview shown; press Enter to accept. |
| **Edge Fill** | W | Edge detection + colour flood-fill. No model needed. Works offline on any image. |
| **SegFormer suggestions** | Button | Pretrained on Cityscapes (road, sidewalk, car, building, sky …). Generates polygon suggestions for the whole image. |
| **Own-model suggestions** | Button | Train a lightweight model on your labels inside the tool, then use it to pre-label the rest. |
| Undo / redo | Ctrl+Z / Ctrl+Y | 100-snapshot history per image. |
| Reviewed flag | R | Mark images as reviewed; filter to unreviewed with Next empty. |
| YOLO-seg labels | — | One `.txt` per image, coordinates normalised 0–1, fully compatible with Ultralytics YOLO. |
| Batch SegFormer | Button | Pre-labels all images in one pass; you review and assign classes. |

---

## 📸 Screenshots

<!-- Add your screenshots here -->
| Smart Select | SegFormer suggestions | Edge Fill |
|:---:|:---:|:---:|
| *(screenshot)* | *(screenshot)* | *(screenshot)* |

---

## 🚀 Quick start

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/RoadAnnoter.git
cd RoadAnnoter

# 2. Install packages
pip install -r requirements.txt

# 3. Download MobileSAM checkpoint (~40 MB)
#    Place it anywhere you like; you will browse to it inside the app.
#    Download link: https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt

# 4. Run
python seg_annotator.py
```

**GPU users:** install PyTorch with CUDA first using the command from
https://pytorch.org/get-started/locally/, then run step 2.

For the full package list with individual pip commands and checkpoint links,
see [REQUIREMENTS.md](REQUIREMENTS.md).

---

## 🗂 Annotation workflow

The recommended workflow for a large image folder:

1. **Generate SegFormer suggestions** on a batch of images
   (Train tab → Generate for ALL images). Requires
   `pip install transformers pillow`.

2. **Review image by image.** Accept suggestions (Enter) or reject (Esc),
   then assign the correct class.

3. **Fix what SegFormer missed** using Smart Select (Q) or Edge Fill (W).

4. **Train your own model** (Train tab → Start training) once you have
   annotated 50–100 images.

5. **Use your trained model** to pre-label the remaining images, review,
   and correct. Repeat.

---

## 🤖 Annotation methods in detail

### Smart Select (Q)
Uses MobileSAM or SAM. Left-click any object — the tool adds a generous box
prompt around the click, runs inference, and picks the largest of the three
candidate masks as the polygon suggestion. Press **Enter** or **Space** to
accept, **Esc** to reject. Right-click to add a background exclusion point.

### Edge Fill (W)
No model required. Left-click inside a region:
1. Bilateral filter + Canny edge detection → barrier walls
2. Colour similarity map (CIELAB) around the click pixel
3. Flood-fill constrained by both barriers and colour
4. Morphological clean-up → polygon

Adjust **Edge detect**, **Edge gap fill** and **Color tolerance** in the panel
to tune for your images.

### SegFormer suggestions
NVIDIA SegFormer fine-tuned on Cityscapes. Available models:

| Model in the dropdown | Hugging Face ID | Approx. size |
|---|---|---|
| SegFormer-B0 (fastest) | `nvidia/segformer-b0-finetuned-cityscapes-512-1024` | ~15 MB |
| SegFormer-B2 (balanced) | `nvidia/segformer-b2-finetuned-cityscapes-1024-1024` | ~110 MB |
| SegFormer-B5 (best) | `nvidia/segformer-b5-finetuned-cityscapes-1024-1024` | ~340 MB |

Models download automatically on first use and are cached locally.
Polygons arrive as class 0 (unassigned); you assign the class yourself.

---

## ⚖ Licence

This tool is released under the **GNU General Public License v3.0** (GPL-3.0).
See [LICENSE](LICENSE).

Third-party models are not redistributed by this repository. Each is subject
to its own licence terms, which the user must follow when downloading and
using those weights.

---

## 📖 Citation

If you use RoadAnnoter in your research, please cite:

```bibtex
@software{haque2026roadannoter,
  author    = {Nazmul Haque},
  title     = {{RoadAnnoter}: A Desktop Annotation Tool for Semantic Segmentation of Road-Scene Images},
  year      = {2026},
  url       = {https://github.com/<your-username>/RoadAnnoter}
}
```

---

## 🙏 Acknowledgements

- [Segment Anything (SAM)](https://github.com/facebookresearch/segment-anything) — Facebook Research
- [MobileSAM](https://github.com/ChaoningZhang/MobileSAM) — Chaoning Zhang et al.
- [SegFormer](https://github.com/NVlabs/SegFormer) — NVIDIA Research
- [Cityscapes Dataset](https://www.cityscapes-dataset.com) — used to train the SegFormer models
