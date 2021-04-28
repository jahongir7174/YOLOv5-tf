[YOLOv5](https://github.com/ultralytics/yolov5) implementation using TensorFlow 2

#### Train
* Change `data_dir`, `image_dir`, `label_dir` and `class_dict` in `config.py`
* Choose version in `config.py`
* Optional, `python main.py --anchor` to generate anchors for your dataset and change anchors in `config.py`
* Optional, `python main.py --record` to generate tf-record for your dataset  
* Run `python main.py --train` for training

#### Test
* Run `python main.py --test`

#### Dataset structure
    ├── Dataset folder 
        ├── images
            ├── 1111.jpg
            ├── 2222.jpg
        ├── labels
            ├── 1111.xml
            ├── 2222.xml
        ├── train.txt
        ├── test.txt
        
#### Note
* xml file should be in PascalVOC format
* `train.txt` `test.txt` contains image names without extension 

#### Recommendation (for docker users)
* `docker pull nvcr.io/nvidia/tensorflow:20.12-tf2-py3`
* `nvidia-docker run --gpus all -v /your/project/folder:/Projects  -it nvcr.io/nvidia/tensorflow:20.12-tf2-py3`
* `cd ../Projects`  
* `apt-get update`
* `apt-get install ffmpeg libsm6 libxext6  -y`
* `pip install opencv-python`

#### Reference
* https://github.com/ultralytics/yolov5
* https://github.com/wizyoung/YOLOv3_TensorFlow