# Output dir
outDir="Images/"
calibDir=${outDir}"Calibration/"

# Input Dir
dataDir="./custom_data/"
obj="bottle"
template=${dataDir}"template.pov"
setting=${dataDir}"setting.pov"

blenderproc run render_calibration.py --root . --obj ${obj}
python3 processMask.py --img_dir ${outDir}
python3 findCorrespondence.py --in_root ${calibDir} --in_dir ${obj} --out_dir ${outDir}

