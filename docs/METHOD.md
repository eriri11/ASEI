# Method and configuration notes

The default configs follow the current ASEI manuscript. The label order used by the model and evidence arrays is background / MYO / LA / LV / AA; result tables use AA / LA / LV / MYO.

## GLFP

Implementation: `asei/glfp.py`; defaults and orchestration: `asei/adaptation.py`.

The frozen prompt teacher predicts an initial largest component and a stable core for each class. Core selection requires agreement under intensity perturbations and confidence of at least 0.9, with at least 16 seed pixels for eligible prompts.

Global boxes scale the class region's bounding box by 1.0, 1.1, and 1.2. Their mean MedSAM response provides global positive and negative evidence. Local boxes begin at core locations and expand along the image axes, subject to seed support, target fraction, size, and overlap constraints. A local response is observed only within its queried box.

Positive responses use a shared threshold of 0.95 for MYO, LA, LV, and AA. Components are retained according to their overlap with the corresponding core or feedback anchor. Global negatives use a response threshold of 0.1. Local negatives also require support from stable background or another organ's original core; accepted local positives are excluded from local negatives.

MYO and LV use local positive evidence as their primary foreground candidates; LA and AA use global positive evidence. Global–local conflicts and pixels supported by multiple class candidates remain unknown. Accepted foreground labels have confidence weights from their supporting responses.

The paper uses **two local rounds**. The first uses the original stable cores; the feedback round uses accepted foreground components touching the original cores to propose additional boxes. Local evidence accumulates, while global evidence is reused. Each class has at most six boxes per round, each with maximum side length 24 pixels. `adaptation.glfp.num_rounds` exposes the total round count for experiments; its paper default is 2.

## ECSS

Implementation: `asei/ecss.py`.

ECSS uses the exact global/local positive/negative masks extracted by GLFP. It selects unknown pixels within 12 pixels of reliable foreground references, excluding GLFP conflicts. A foreground class is excluded only where negative evidence is present without positive evidence; background remains allowed. Pixels with a nontrivial allowed class set receive the loss `-log(sum(p[c] for c in allowed_classes))`.

The initial pixel weight is 0.1. Implementation controls on class-set size and auxiliary mass are specified in `adaptation.ecss`; sets are aligned with image augmentation before loss computation. ECSS shares the prepared MedSAM responses with GLFP.

## Joint supervision

Implementation: `asei/adaptation.py`, `asei/_losses.py`, and `asei/_spatial.py`.

- **H:** hard pseudo-labels with confidence-weighted CE and foreground Dice losses.
- **C:** ECSS class-set supervision.
- **M:** remaining unknown pixels with student/EMA-teacher probability consistency.

The CE, foreground Dice, ECSS, and consistency loss coefficients are all 1 in both adaptation directions.

The three masks are disjoint. GLFP supervision and ECSS class sets are prepared before the final adaptation stage and remain fixed. The prompt teacher and MedSAM are frozen. The EMA teacher is updated during adaptation and used for evaluation.

## Outputs and caches

The output directory contains the resolved `config.yaml`, `prompt_teacher.pt`, `preparation.json`, `student_final.pt`, `teacher_final.pt`, `summary.json`, and training logs. Source training additionally saves `source_split.json` and validation history.

MedSAM image embeddings, GLFP response/label banks, and ECSS class sets use separate content-addressed caches. Input hashes, construction options, and implementation hashes control reuse. Changing only final-stage training settings can reuse compatible prepared supervision. Changes to evidence construction regenerate affected banks. Training outputs are written under `runs/`, which is ignored by version control.
