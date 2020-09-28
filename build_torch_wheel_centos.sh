#!/bin/bash

#set -e  # Fail on any error.
set -x  # Display commands being run.

PYTHON_VERSION=$1
RELEASE_VERSION=$2  # rX.Y or nightly
DEFAULT_PYTHON_VERSION=3.6
DEBIAN_FRONTEND=noninteractive
CONDA_VERSION=5.3.1

function install_cmake() {
   wget https://cmake.org/files/v3.6/cmake-3.6.2.tar.gz
   tar xzvf cmake-3.6.2.tar.gz
   cd cmake-3.6.2
   ./bootstrap --prefix=/usr/local
   make
   make install
   source ~/.bash_profile
}

function install_and_setup_conda() {
     curl -O "https://repo.anaconda.com/archive/Anaconda3-${CONDA_VERSION}-Linux-x86_64.sh"
     sh "Anaconda3-${CONDA_VERSION}-Linux-x86_64.sh" -b
     rm -f "Anaconda3-${CONDA_VERSION}-Linux-x86_64.sh"
     source /root/anaconda3/etc/profile.d/conda.sh
     ENVNAME="pytorch"
     if conda env list | awk '{print $1}' | grep "^$ENVNAME$"; then     conda remove --name "$ENVNAME" --all;   fi
     if [ -z "$PYTHON_VERSION" ]; then     PYTHON_VERSION=$DEFAULT_PYTHON_VERSION;   fi
     conda create -y --name "$ENVNAME" python=${PYTHON_VERSION} anaconda
     conda activate "$ENVNAME"
     export CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"
     conda install -y numpy pyyaml mkl-include setuptools cmake cffi typing tqdm coverage tensorboard hypothesis dataclasses
     /usr/bin/yes | pip install --upgrade google-api-python-client
     /usr/bin/yes | pip install --upgrade oauth2client
     /usr/bin/yes | pip install --upgrade google-cloud-storage
     /usr/bin/yes | pip install lark-parser
     /usr/bin/yes | pip install cloud-tpu-client
     /usr/bin/yes | pip install tensorboardX
 }

function install_llvm_clang {
    install_cmake
    git clone https://github.com/llvm/llvm-project -b release/8.x
    cd llvm-project
    mkdir build
    cd build/
    cmake -G "Unix Makefiles" -DLLVM_ENABLE_PROJECTS="clang"  -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=/opt/rh/devtoolset-9/root/usr/bin/gcc -DCMAKE_CXX_COMPILER=/opt/rh/devtoolset-9/root/usr/bin/g++ ../llvm
    cmake --build .
    export CPATH=/root/anaconda3/include:$CPATH
    make -j
    make install
}

function maybe_append {
  local LINE="$1"
  local FILE="$2"
  if [ ! -r "$FILE" ]; then
    if [ -w "$(dirname $FILE)" ]; then
      echo "$LINE" > "$FILE"
    else
      sudo bash -c "echo '$LINE' > \"$FILE\""
    fi
  elif [ "$(grep -F "$LINE" $FILE)" == "" ]; then
    if [ -w "$FILE" ]; then
      echo "$LINE" >> "$FILE"
    else
      sudo bash -c "echo '$LINE' >> \"$FILE\""
    fi
  fi
}

function setup_system {
  if [ "$CXX_ABI" == "0" ]; then
    export CFLAGS="${CFLAGS} -D_GLIBCXX_USE_CXX11_ABI=0"
    export CXXFLAGS="${CXXFLAGS} -D_GLIBCXX_USE_CXX11_ABI=0"
  fi
}

function install_cudnn {
  if [ "$CUDNN_TGZ_PATH" == "" ]; then
    echo "Missing CUDNN_TGZ_PATH environment variable"
    exit 1
  fi
  local CUDNN_FILE="cudnn.tar.gz"
  if [[ "$CUDNN_TGZ_PATH" == gs://* ]]; then
    gsutil cp "$CUDNN_TGZ_PATH" "$CUDNN_FILE"
  elif [[ "$CUDNN_TGZ_PATH" == http* ]]; then
    wget -O "$CUDNN_FILE" "$CUDNN_TGZ_PATH"
  else
    ln -s "$CUDNN_TGZ_PATH" "$CUDNN_FILE"
  fi
  tar xvf "$CUDNN_FILE"
  sudo cp cuda/include/cudnn.h /usr/local/cuda/include
  sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64
  sudo chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda/lib64/libcudnn*
  rm -rf cuda
  rm -f "$CUDNN_FILE"
}

function maybe_install_cuda {
  if [ "$XLA_CUDA" == "1" ]; then
    if [ ! -d "/usr/local/cuda" ]; then
      local CUDA_VER="10.2"
      local CUDA_SUBVER="89_440.33.01"
      local CUDA_FILE="cuda_${CUDA_VER}.${CUDA_SUBVER}_linux.run"
      wget "http://developer.download.nvidia.com/compute/cuda/${CUDA_VER}/Prod/local_installers/${CUDA_FILE}"
      sudo sh "${CUDA_FILE}" --silent --toolkit
      rm -f "${CUDA_FILE}"
    fi
    if [ ! -f "/usr/local/cuda/include/cudnn.h" ] && [ ! -f "/usr/include/cudnn.h" ]; then
      install_cudnn
    fi
    export TF_CUDA_PATHS="/usr/local/cuda,/usr/include,/usr"
    maybe_append 'export TF_CUDA_PATHS="/usr/local/cuda,/usr/include,/usr"' ~/.bashrc
    if [ "$TF_CUDA_COMPUTE_CAPABILITIES" == "" ]; then
      export TF_CUDA_COMPUTE_CAPABILITIES="7.0"
    fi
    maybe_append "export TF_CUDA_COMPUTE_CAPABILITIES=\"$TF_CUDA_COMPUTE_CAPABILITIES\"" ~/.bashrc
  fi
}

function maybe_install_sources {
  if [ ! -d "torch" ]; then
    sudo apt-get install -y git
    git clone --recursive https://github.com/pytorch/pytorch.git
    cd pytorch
    git clone --recursive https://github.com/pytorch/xla.git
    export RELEASE_VERSION="nightly"
  fi
}

function install_bazel() {
     wget https://copr.fedorainfracloud.org/coprs/vbatts/bazel/repo/epel-7/vbatts-bazel-epel-7.repo
     mv vbatts-bazel-epel-7.repo /etc/yum.repos.d/
     yum install -y bazel3
}
function install_req_packages() {
     yum install -y wget
     yum install -y openssl-devel bzip2-devel
     maybe_install_cuda
     install_bazel
}


function install_gcloud() {
  if [ "$(which gcloud)" == "" ]; then
      curl https://sdk.cloud.google.com > install.sh
      bash install.sh --disable-prompts
  fi
}

function install_and_setup_conda() {
  # Install conda if dne already.
  if ! test -d "$HOME/anaconda3"; then
    CONDA_VERSION="5.3.1"
    curl -O "https://repo.anaconda.com/archive/Anaconda3-${CONDA_VERSION}-Linux-x86_64.sh"
    sh "Anaconda3-${CONDA_VERSION}-Linux-x86_64.sh" -b
    rm -f "Anaconda3-${CONDA_VERSION}-Linux-x86_64.sh"
  fi
  maybe_append ". $HOME/anaconda3/etc/profile.d/conda.sh" ~/.bashrc
  source "$HOME/anaconda3/etc/profile.d/conda.sh"
  ENVNAME="pytorch"
  if conda env list | awk '{print $1}' | grep "^$ENVNAME$"; then
    conda remove --name "$ENVNAME" --all
  fi
  if [ -z "$PYTHON_VERSION" ]; then
    PYTHON_VERSION=$DEFAULT_PYTHON_VERSION
  fi
  conda create -y --name "$ENVNAME" python=${PYTHON_VERSION} anaconda
  conda activate "$ENVNAME"
  export CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

}

function build_and_install_torch() {
  # Checkout the PT commit ID or branch if we have one.
  COMMITID_FILE="xla/.torch_pin"
  if [ -e "$COMMITID_FILE" ]; then
    git checkout $(cat "$COMMITID_FILE")
  fi
  # Only checkout dependencies once PT commit/branch checked out.
  git submodule update --init --recursive
  # Apply patches to PT which are required by the XLA support.
  xla/scripts/apply_patches.sh
  python setup.py bdist_wheel
  pip install dist/*.whl
}

function build_and_install_torch_xla() {
  git submodule update --init --recursive
  if [ "${RELEASE_VERSION}" = "nightly" ]; then
    export VERSIONED_XLA_BUILD=1
  else
    export TORCH_XLA_VERSION=${RELEASE_VERSION:1}  # r0.5 -> 0.5
  fi
  python setup.py bdist_wheel
  pip install dist/*.whl
}

function install_torchvision_from_source() {
  torchvision_repo_version="master"
  # Cannot install torchvision package with PyTorch installation from source.
  # https://github.com/pytorch/vision/issues/967
  git clone -b "${torchvision_repo_version}" https://github.com/pytorch/vision.git
  pushd vision
  python setup.py bdist_wheel
  pip install dist/*.whl
  popd
}

function main() {
  setup_system
  maybe_install_sources
  install_req_packages
  install_llvm_clang
  install_and_setup_conda
  build_and_install_torch
  pushd xla
  build_and_install_torch_xla
  popd
  install_torchvision_from_source
  install_gcloud
}

#main
