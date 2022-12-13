blenderproc run render_calibration.py --root . --obj bottle
python findCorrespondence.py --in_root images/Calibration --in_dir bottle --out_dir images

