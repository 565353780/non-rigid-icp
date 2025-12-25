conda install -c conda-forge scikit-sparse -y

pip3 install torch torchvision \
  --index-url https://download.pytorch.org/whl/cu128

pip install open3d numpy scipy tqdm scikit-image \
  scikit-learn tensorboard opencv-python trimesh \
  ninja

pip install moviepy==1.0.3
