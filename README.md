# DeepLearning Models with Pytorch

Pytorch로 직접 구현한 모델입니다.   
공부용으로 잘못된 부분이 존재할 수 있습니다.   

## 작업환경

#### Docker

```
# image : nvcr.io/nvidia/pytorch:21.10-py3
docker run -itd --name pytorch_model --gpus all --net=host --ipc=host -v $(pwd)/../pytorch_models:/github nvcr.io/nvidia/pytorch:21.10-py3
```
