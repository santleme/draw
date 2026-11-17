# Alternative hand sources — in case FreiHAND does not have a suitable mask

> Scope of this document: **only** hand sources for cutout and
> insertion into synthetic scenes of the Draw neural project
> (segmentation 0=background, 1=paper, 2=hand, 3=ink, visible rule:
> hand over stroke = `hand`).
>
> This document does **not** change the main plan. It exists for the
> case where the FreiHAND audit (sibling document
> `DATASET_FREIHAND.md`, another agent) concludes that the Kaggle copy
> `danieldelro/freihand` contains no usable masks.
>
> Honesty rule applied here: a license is only asserted when
> found in an official source for the dataset. Everything else is marked
> as **NOT CONFIRMED** — do not use the source in production before
> confirming the license in the official source.
>
> No whole dataset was downloaded for this research. Verification
> was done by file/metadata listing on Kaggle and by web search.

## 0. Starting point: what official FreiHAND promises

According to the official FreiHAND description (Kaggle, copy
`danieldelro/freihand`, Freiburg University):

- 32,560 unique training samples, post-processed into 4 versions
  (green screen / background / sample / automatic) = 130,240 training
  images, plus 3,960 evaluation images.
- Each training sample promises: **224×224** RGB image, **224×224 hand
  segmentation mask**, intrinsic matrix K, hand scale,
  21 3D keypoints and 3D shape.
- Masks exist **only for training**; in evaluation, keypoints and
  shape are withheld.
- Terms stated on the page: **research only, commercial use
  prohibited, no warranty; citing the paper is mandatory**
  (Zimmermann et al., ICCV 2019).

Consequence: if the Kaggle copy does in fact contain the
`training/mask/` folder (expected format: `training/mask/{id:08d}.jpg`,
binary mask, no version — see utility `fh_utils.read_msk`),
**there is no need for an alternative mask source**.
The alternatives below only come in if that verification fails, or as
a diversity complement — and even then with the listed limitations.

Limitations of FreiHAND that hold even with a mask (reason a
backup exists):

- Centered hand, 224×224, originating from green screen with composited backgrounds:
  not the Draw viewpoint (top-down camera, hand entering from the
  edge over an A4 sheet).
- Mask in JPG: edge with compression artifacts; for cutout,
  threshold (e.g. > 127) and, if needed, close morphologically.
- Low resolution for the model's 512×512 input: upscale 224 → 512
  blurs the hand edge; test whether this harms the `hand` class.

## 1. Comparison table

| Source | Ready mask? | Format / size | Viewpoint | License / terms | Cutout viability for Draw | Verdict |
|---|---|---|---|---|---|---|
| **FreiHAND (main)** | Yes, per official description (training) | RGB + binary mask 224×224; Kaggle 3.9 GB; 130,240 training / 3,960 evaluation | Centered hand, green screen | Research only, commercial prohibited, cite paper | High, if the `training/mask/` folder exists in the Kaggle copy; threshold JPG, upscale with care | **1st choice — keep** |
| **Render mask from FreiHAND 3D mesh** | No — it is a derived mask | 3D vertices + K per sample; render mesh projection | Identical to FreiHAND (same images) | Same as FreiHAND (research only) | Medium: requires projecting mesh with K, handling finger-finger occlusion and edge; risk of diverging from official mask; only worthwhile if official mask is missing | **Fallback, do not prefer** |
| **EgoHands (Indiana, Bambach et al., ICCV 2015)** | Yes, but as polygons, not PNG | 4,800 images, > 15,000 hand instances; 48 locations, 100 images each; annotation in `polygons.mat` per location; needs polygon rasterization → binary mask | Egocentric, Google Glass, activities: puzzle, chess, Jenga, cards | **NOT CONFIRMED** in the original source (the CC BY 4.0 license seen is from the mirror copy on Roboflow Universe, not the original page — do not use as proof) | High for cutout, after rasterization; real environment and lighting diversity is the best complement to FreiHAND; note: other people's hands, social interaction, not a drawing hand | **Backup no. 1** |
| **RHD — Rendered Handpose Dataset (Zimmermann & Brox, 2017)** | Yes, by class | 41,258 training + 2,728 test, 320×320; masks by: background, person, 3 classes per finger + palm; RGB + depth + 21 keypoints | Synthetic, Mixamo characters rendered in Blender, random background | Research only, commercial prohibited, cite paper (terms from the RHD Kaggle page) | High, after collapsing finger+palm → binary `hand`; clean mask, no JPG artifact; risk: synthetic domain (texture/skin) different from real hand | **Backup no. 2 (augmentation, not sole source)** |
| **GTEA / EGTEA (Georgia Tech)** | Yes, pixel by pixel | GTEA: 663 images with pixel annotation (or 625 with two-hand label + 38 binary, depending on the source); EGTEA: 12,799 binary + 1,048 two-hand; 7 activities, 4 subjects, kitchen | Egocentric, head-mounted, kitchen, static lighting | **NOT CONFIRMED** | Medium-low: little environment/lighting and subject diversity; useful only as a small complement | **Small complement, not main backup** |
| **EYTH — EgoYouTubeHands** | Yes, pixel by pixel | ~2,000 frames, ~2,600 instances (~1,800 first person + ~800 third person), from only 3 videos | YouTube, in-the-wild, diverse | **NOT CONFIRMED**; YouTube origin brings extra redistribution risk | Medium: real diversity, but frames very similar to each other (fixed sampling) and few videos — high leak risk between splits if not separated by video | **Complement, separate by video** |
| **Ego2Hands (Lin et al., 2021)** | Yes (energy/segmentation, two hands) | Green-screen composition of right hand + mirror as left; evaluation 800×448, 8 sequences | Egocentric, two hands, free movement | **NOT CONFIRMED** | Medium: only option dedicated to **two hands**; green-screen composition may leave edge artifacts; use only for the generator's two-hand scenes | **Only for two-hand scenes** |
| **HandOverFace (HOF)** | Yes | Hands over face, in-the-wild | Third person, hand over face | **NOT CONFIRMED** | Low for Draw: wrong viewpoint and context (face, not sheet) | **Do not use** |
| **SynthHands (MPI, ICCV 2017)** | Mask via Unity rendering (not simple PNG) | 63,530 pose frames, 5 egocentric viewpoints, 48.5 GB in a single zip | Synthetic egocentric, Unity, RealSense SR300 | Scientific/non-commercial only, cite paper (license inside the download) | Low: 48.5 GB, Unity pipeline, and the whole download is disproportionate to the gain versus RHD/FreiHAND | **Do not use** |
| **AnyHand (synthetic, 2025)** | Yes (RGB, depth, mask, bbox, pose) | 1.05M single-hand scenes / 2.1M interact scenes, 47,438 shapes, 10,240 textures | Synthetic, multiple cameras | **NOT CONFIRMED** (size and license not verified in an official source in this research) | Not assessable without confirming access/license/size; do not base the plan on it | **Do not use until confirmed** |
| **HaDR (synthetic, domain randomization)** | Yes, RGB-D + mask | 117 thousand samples (per literature comparison) | Synthetic, manipulation | **NOT CONFIRMED** | Not assessable without confirming access/license | **Do not use until confirmed** |
| **LSWH100 (Libras, Zenodo)** | Yes (mask + depth, COCO) | 144,000 images (48,000 RGB × 3 sets), 100 hand shapes, Blender, Unicamp | Centered hand, signing, 4 perspectives | **NOT CONFIRMED** (Zenodo record 10.5281/zenodo.10635628 — check license in the record before using) | Low for Draw: Libras shapes, centered hand, not a drawing/writing hand | **Do not use for Draw** |
| **Hands captured by Draw itself (future)** | To annotate | To be defined by Codex/user | **Exactly Draw's**: top-down camera, hand over sheet | Internal, owner's consent | Ideal long-term, but outside Muse's scope (do not capture camera) | **Long-term, not now** |

## 2. Recommendation

1. **First choice: FreiHAND direct.** If `training/mask/` exists
   in the Kaggle copy, use RGB + binary mask, threshold, and record
   in the dataset manifest: origin, version, license (research only)
   and the viewpoint/resolution limitation.
2. **Backup no. 1: EgoHands.** If FreiHAND fails, or for real
   diversity beyond FreiHAND: rasterize `polygons.mat` → binary PNG
   mask, separate splits **by location/video** (never by frame), and
   **confirm the license on Indiana's original page before
   publishing any derivative**. Kaggle access verified in this
   research (without downloading everything): `himaniishah/egohands` (1.33 GB),
   `chaturvediabhay24/egohandssegmentation` (189 MB) — note:
   these are third-party re-uploads, not the official source; prefer the
   official source when possible.
3. **Backup no. 2: RHD, only as augmentation.** Clean and large mask
   (41 thousand), but synthetic: mix with real hand, never train only
   on it, so as not to learn synthetic texture as `hand`.
4. **Rendered mesh: only if the official mask is missing.** Rendering the
   FreiHAND mesh with K is technically viable, but it is extra
   work to reproduce what the official mask already delivers — and it may
   diverge from it at the edge. If that is the path, validate the rendered
   mask against a visual sample before generating the dataset.
5. **Two hands: Ego2Hands, only for that case.** Do not use as the main
   single-hand source.

## 3. Cutout viability (what the generator needs, per source)

For any source, the synthetic generator needs, per sample:

- Hand RGB (or whole frame) + binary mask **of the same size**;
- single conversion: `hand_mask = (mask > threshold)`; for RHD,
  `hand_mask = (class ∈ {fingers, palm})`; for EgoHands,
  `hand_mask = rasterize(polygons)`; for Ego2Hands, use the
  segmentation provided, not the detection energy;
- cutout with soft edge (1–2 px feather) so as not to leave a green/white
  green-screen halo;
- provenance record per sample (source, original id, split),
  so the split by source does not leak the same hand between train/val/test.

## 4. What remains to confirm (before using any backup)

- [ ] FreiHAND in the Kaggle copy: does `training/mask/` exist? Exact format
      (binary JPG? values 0/255?) — the dataset agent's job.
- [ ] Original EgoHands (Indiana): license/terms of use on the
      official page; current official download link.
- [ ] GTEA/EGTEA, EYTH, Ego2Hands: official license of each.
- [ ] RHD: confirm whether the "research only" license allows generating
      derived synthetic compositions for Draw training (not just
      reproducing the dataset).
- [ ] Any source with identifiable people: check whether there is
      a restriction on redistribution of derived images (cutouts),
      not just of the original dataset.
- [ ] No source from this list should be published/redistributed
      together with the model without that confirmation — Draw is a project
      with future commercial potential, and "research only" does not cover that.

## 5. Sources consulted

- FreiHAND, Kaggle `danieldelro/freihand` (official description:
  RGB 224×224 + mask 224×224 in training, research only, cite
  Zimmermann et al. 2019) — verified via Kaggle listing
  (`datasets list --search freihand`, partial `datasets files`).
- FreiHAND utils (`fh_utils.read_msk`: `training/mask/{id:08d}.jpg`,
  binary mask, training only).
- EgoHands: Bambach et al., ICCV 2015; 4,800 images, > 15,000
  instances, `polygons.mat`; Kaggle: `himaniishah/egohands`,
  `chaturvediabhay24/egohandssegmentation`.
- GTEA/EGTEA: 663 pixel images (GTEA) / 12,799 binary + 1,048
  two-hand (EGTEA); kitchen, 4 subjects.
- EYTH: ~2,000 frames / ~2,600 instances, 3 videos.
- RHD: Zimmermann & Brox 2017; 41,258 training + 2,728 test, 320×320,
  masks by finger/palm; research only, cite paper.
- SynthHands: MPI ICCV 2017; 48.5 GB, scientific/non-commercial.
- Ego2Hands: Lin et al. 2021; green-screen composition, evaluation
  800×448.
- AnyHand / HaDR / LSWH100: sizes and licenses **not confirmed**
  in an official source in this research — listed only as candidates
  discarded for now.
