# Toy problem: Deep Directly-Trained Spiking Neural Networks for Object Detection

## Model hypthesis for the self created dataset

    The aim of this control experiment is to classify clockwise(CW) motion and counter clockwise(CCW) motion. The only controled difference of the dataset is the motion orientation. In the self maded dataset, tempol order is the only reliable difference. Whether the EMS-YOLO model could detect the correct motion orientation is the control experiment tryin to figure out.

### The goal of this dataset:

    Temporal pulse of EMS-YOLO can use the temporal order of the event to detect or classify the objects.

### Positive result:

    For those events with same space content but reversed time order, the EMS-YOLO can successfully detect the clockwise(CW) and counter clockwise(CCW) motions. But when the model compress the time order, for example use T=1 instead of T=5, the detecting performance of EMS-YOLO model will decrease.

### Dataset and model

Dataset:

* The dataset should ensure that the space contents of CW and CCW.
* The start point, end point, bounding box, number of events are the same.
* The only reliable difference is the temporal order of the five event bins.

Model:
    Can EMS-YOLO use this unique difference to perform orientation classification detection?

### Core control

* Catagory 0: Clockwise motion
* Catagory 1:Counter clockwise motion
* The object will back to the same end point every 5 time step.
* The CW and CCW dataset use 5 same event frame
* Same position of the final bounding box

Therefore, the dataset should have strong control: the catagory can only be classified by the temporal order, could not leak from the final position, background, speed or event amounts.

### Two dataset conditions

**Control data**

* Different shape of the objects. CW is circle, CCW is square.
* Can be classified when the temporal information is lost
* used to prove the reliability of the model

**Temporal_only data**

* Same shape of objects for both CW and CCW,circle
* Only difference is the temporal order
* If the temporal order is compressed, the model will have nearly random classification performance theoretically

Apart from the different temporal order, the number of samples, trajectory, location, background, amount of events, target size, and label distribution are all the same.

## Dataset creation
