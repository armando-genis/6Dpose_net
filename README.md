# 6Dpose_net


pip install --upgrade git+https://github.com/aleju/imgaug.git


   31  pip install opencv-python\n
   32  python3 models/model.py
   33  pip install pyximport
   34  pip install cython\n
   35  pip install pyximport

   python3 test_train.py --phi 0 --epochs 600 --validation-image-save-path /workspace/eval_images/ linemod /workspace/dataset-custom-v10-ladle-big/ --object-id 


   python3 train.py --phi 0 --weights /workspace/network_tensortflow/phi_0_linemod_best_ADD.h5 --epochs 600 --validation-image-save-path /workspace/eval_images/ linemod /workspace/dataset-custom-v10-ladle-big/ --object-id 1

   python3 train.py --phi 0 --weights imagenet --epochs 600 --validation-image-save-path /workspace/eval_images/ linemod /workspace/dataset-custom-v3-ladle/ --object-id 1

   