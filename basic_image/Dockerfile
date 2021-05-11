FROM tensorflow/tensorflow:latest-gpu-jupyter
ADD ./ /pip_requirement
WORKDIR /pip_requirement
RUN pip install -U -r /pip_requirement/requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
