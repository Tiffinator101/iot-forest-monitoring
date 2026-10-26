# Development scripts

The steps that led to the final image processing method. They are kept for reference and are
not the code the report describes. The final code is in the folder above:
`tree_measure.py` (the method), `app.py` (the Tree Measurement app) and
`step7_closeup_trunk.py` (the close-up trunk test).

- `step2_coco_test.py`: standard YOLO11 (COCO), which has no tree class.
- `step3_yoloe_tree.py`: YOLOE with the prompt "tree".
- `step4_height_methods.py`, `step4b_preprocessing.py`, `step4c_prompts_models.py`: attempts to get
  the trunk base from YOLOE (masks, image changes, prompts and bigger models).
- `step5_crown_trunk.py`: YOLOE crown plus classical trunk detection, in pixels.
- `step6_measure_metres.py`: pixels to metres with the stick-calibrated focal length. Needs step 5's CSV.
- `simple_measure.py`: steps 5 and 6 simplified into one script, trunk width at 0.5 m.
  `tree_measure.py` is this method as one function.

Run them with the `.venv` in the `ImageProcessing` folder, for example
`.venv\Scripts\python.exe development\step5_crown_trunk.py`. They load their models from `models/`
and their results still go to `output/`.
