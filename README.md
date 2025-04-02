# 6Dpose_net
THis is an un-finish 6d pose detection. 

pip install --upgrade git+https://github.com/aleju/imgaug.git
pip install opencv-python
pip install cython
pip install pyyaml
pip install plyfile
pip install progressbar2
pip install typeguard
pip install tqdm


 ### to run network_imu

   python3 test_train.py --phi 0 --epochs 600 --validation-image-save-path /workspace/eval_images/ linemod /workspace/dataset-custom-v10-ladle-big/ --object-id 

 ### to run network_tensorflow

   python3 train.py --phi 0 --weights /workspace/network_tensortflow/phi_0_linemod_best_ADD.h5 --epochs 600 --validation-image-save-path /workspace/eval_images/ linemod /workspace/dataset-custom-v10-ladle-big/ --object-id 1

   python3 train.py --phi 0 --weights imagenet --epochs 600 --validation-image-save-path /workspace/eval_images/ linemod /workspace/dataset-custom-v10-ladle-big/ --object-id 1

   python3 train.py --phi 0 --weights imagenet --epochs 600 --gpu 0 --validation-image-save-path /workspace/eval_images/ linemod /workspace/dataset-custom-v10-ladle-big/ --object-id 1

   python3 train.py --phi 0 --weights /workspace/network_tensortflow/phi_0_linemod_best_ADD.h5 --epochs 600 --gpu 0 --validation-image-save-path /workspace/eval_images/ linemod /workspace/dataset-custom-v10-ladle-big/ --object-id 1


python3 train.py --phi 0 --weights /workspace/network_tensortflow/attention_checpoint.h5 --epochs 600 --gpu 0 --validation-image-save-path /workspace/eval_images/ linemod /workspace/dataset-custom-v10-ladle-big/ --object-id 1


python3 train.py --phi 0 --weights imagenet --epochs 600 --gpu 0 --validation-image-save-path /workspace/eval_images/ linemod /workspace/dataset-custom-v10-ladle-big/ --object-id 1