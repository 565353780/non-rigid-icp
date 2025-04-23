if [ $(uname) = "Darwin" ]; then
  brew install suite-sparse
fi
if [ $(uname) = "Linux" ]; then
  sudo apt-get install libsuitesparse-dev
fi

pip install -U torch torchvision torchaudio

pip install -U open3d numpy scipy tqdm scikit-image \
  scikit-learn scikit-sparse tensorboard opencv-python \
  trimesh ninja

pip install moviepy==1.0.3
