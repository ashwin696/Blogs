# Serving and Managing ML models with Mlflow and Triton Inference Server  

## Lets get brief familiarity with Mlflow, what it is and what it offers.
**Introduction:**

MLflow is an open source platform for managing the end-to-end machine learning lifecycle. It tackles four primary functions:

- Tracking experiments to record and compare parameters and results ([MLflow Tracking](<https://mlflow.org/docs/latest/tracking.html#tracking>)).

- Packaging ML code in a reusable, reproducible form in order to share with other data scientists or transfer to production ([MLflow Projects](<https://mlflow.org/docs/latest/projects.html#projects>)).

- Managing and deploying models from a variety of ML libraries to a variety of model serving and inference platforms ([MLflow Models](<https://mlflow.org/docs/latest/models.html#models>)).

- Providing a central model store to collaboratively manage the full lifecycle of an MLflow Model, including model versioning, stage transitions, and annotations ([MLflow Model Registry](<https://mlflow.org/docs/latest/model-registry.html#registry>)).

![alt text](imgs/mlflow.jpg "Mlfow Architecture")

**Installation:**

Install mlflow in python env

```
pip install mlflow # includes UI
pip install mlflow[extras] # downloads extra ML libraries
```

### [MLflow on ](<https://mlflow.org/docs/latest/tracking.html#id29>)[localhost](<http://localhost>)[ with SQLite](<https://mlflow.org/docs/latest/tracking.html#id29>):

```
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./artifacts --host 0.0.0.0 --port 5000
```


![alt text](imgs/sqlite.png "Backend SQlite")  

<u>Backend Store </u>: To store Run/Experiment information along with Model Store info as db.  

<u>Artifacts</u>: Output files of Runs/Experiements (Models, conifgs, labels etc)



For Other Types of Tracking methods [check here](<https://mlflow.org/docs/latest/tracking.html#id27>).



**Start Logging runs/experiments using Python API's:**

### [Logging Functions](<https://mlflow.org/docs/latest/tracking.html#id59>)

[`mlflow.set_tracking_uri()`](<https://mlflow.org/docs/latest/python_api/mlflow.html#mlflow.set_tracking_uri>) connects to a tracking URI. You can also set the `MLFLOW_TRACKING_URI` environment variable to have MLflow find a URI from there. In both cases, the URI can either be a HTTP/HTTPS URI for a remote server, a database connection string, or a local path to log data to a directory. The URI defaults to `mlruns`.

[`mlflow.create_experiment()`](<https://mlflow.org/docs/latest/python_api/mlflow.html#mlflow.create_experiment>) creates a new experiment and returns its ID. Runs can be launched under the experiment by passing the experiment ID to `mlflow.start_run`.

[`mlflow.set_experiment()`](<https://mlflow.org/docs/latest/python_api/mlflow.html#mlflow.set_experiment>) sets an experiment as active. If the experiment does not exist, creates a new experiment. If you do not specify an experiment in [`mlflow.start_run()`](<https://mlflow.org/docs/latest/python_api/mlflow.html#mlflow.start_run>), new runs are launched under this experiment.

[`mlflow.start_run()`](<https://mlflow.org/docs/latest/python_api/mlflow.html#mlflow.start_run>) returns the currently active run (if one exists), or starts a new run and returns a [`mlflow.ActiveRun`](<https://mlflow.org/docs/latest/python_api/mlflow.html#mlflow.ActiveRun>) object usable as a context manager for the current run. You do not need to call `start_run` explicitly: calling one of the logging functions with no active run automatically starts a new one.

[`mlflow.end_run()`](<https://mlflow.org/docs/latest/python_api/mlflow.html#mlflow.end_run>) ends the currently active run, if any, taking an optional run status.

[`mlflow.log_metric()`](<https://mlflow.org/docs/latest/python_api/mlflow.html#mlflow.log_metric>) logs a single key-value metric. The value must always be a number. MLflow remembers the history of values for each metric. Use [`mlflow.log_metrics()`](<https://mlflow.org/docs/latest/python_api/mlflow.html#mlflow.log_metrics>) to log multiple metrics at once.

View the dashboard at [http://localhost:5000](<http://localhost:5000>) where the mlflow server is running.

![Backend SQlite](imgs/mlflow_dash.png "Backend SQlite")

# [MLflow Model Registry:](<https://mlflow.org/docs/latest/model-registry.html#api-workflow>)

The MLflow Model Registry component is a centralized model store, set of APIs, and UI, to collaboratively manage the full lifecycle of an MLflow Model. It provides model lineage (which MLflow experiment and run produced the model), model versioning, stage transitions (for example from staging to production), and annotations.

Adding the model to Model Registry:

```python
from random import random, randint
from sklearn.ensemble import RandomForestRegressor

import mlflow
import mlflow.sklearn

with mlflow.start_run(run_name="YOUR_RUN_NAME") as run:
    params = {"n_estimators": 5, "random_state": 42}
    sk_learn_rfr = RandomForestRegressor(**params)

    # Log parameters and metrics using the MLflow APIs
    mlflow.log_params(params)
    mlflow.log_param("param_1", randint(0, 100))
    mlflow.log_metrics({"metric_1": random(), "metric_2": random() + 1})

    # Log the sklearn model and register as version 1
    mlflow.sklearn.log_model(
        sk_model=sk_learn_rfr,
        artifact_path="sklearn-model",
        registered_model_name="sk-learn-random-forest-reg-model"
    )
```

More features like Fetching Serving, Renaming etc. [https://mlflow.org/docs/latest/model-registry.html#id6](<https://mlflow.org/docs/latest/model-registry.html#id6>)  


# Now lets see what Triton Inference Server is,

<b>Introduction</b>:

NVIDIA Triton Inference Server is an open source inference serving software that streamlines AI inferencing. Triton enables teams to deploy any AI model from multiple deep learning and machine learning frameworks, including TensorRT, TensorFlow, PyTorch, ONNX, OpenVINO, Python, RAPIDS FIL, and more.   
Triton supports inference across cloud, data center,edge and embedded devices on NVIDIA GPUs, x86 and ARM CPU, or AWS Inferentia.

### <strong>Serving with Triton Inference Server with mlflow:</strong>  

For serving models with Triton inference Server, Nvidia provides Mlflow Triton Plugin.


Currently it supports onnx and triton model flavours.

**Steps:**

1. Run Triton Inference Server

2. Run MLFlow Triton Plugin

3. Publish Models to Mlflow server

4. Deploy the published models to Triton



Start Triton Inference Server in <u>EXPLICIT</u> mode

```bash
docker run --gpus=1 --rm -p8000:8000 -p8001:8001 -p8002:8002 -v/home/ubuntu/triton_models:/models nvcr.io/nvidia/tritonserver:22.12-py3 tritonserver --model-repository=/models --model-control-mode=explicit

# Explicit mode does not load models at runtime.
```



**Mlfow Triton Plugin**:

Create a folder in the machine with name <u>triton_models</u> and copy your models to this folder with model structure as required by Triton Server

**Model Structure for Inferencing:**

```
└── model_folder/      # model_folder
    ├── 1              # Version of the model
        └── model.ckpt # model file
    ├── config.pbxt    # model configfile
    └── labels.txt     # labels of classes
```

Here we take an example of yolov6n:
```
└── yolov6n/      # model_folder
    ├── 1              # Version of the model
        └── model.onnx # model file
    ├── config.pbxt    # model configfile
    └── labels.txt     # labels of classes
```

Create MLFlow Triton Plugin container with volume mount to Triton model repository and open bash in the container:

```bash
docker run -it -v /home/ubuntu/triton_models:/triton_models \
--env TRITON_MODEL_REPO=/triton_models \
--gpus '"device=0"' \
--net=host \
--rm \
-d nvcr.io/nvidia/morpheus/mlflow-triton-plugin:2.2.2

docker exec -it <container_name> bash
```

Export the Mlflow tracking server url and start the server: 
```bash
export MLFLOW_TRACKING_URI=http://localhost:5000

nohup mlflow server --backend-store-uri sqlite:////tmp/mlflow-db.sqlite --default-artifact-root /mlflow/artifacts --host 0.0.0.0 &
```

Publish reference models to MLflow:

```bash
python publish_model_to_mlflow.py --model_name yolov6n  --model_directory /triton_models/yolov6n --flavor triton
```
![Model Dashboard](imgs/mlflow_models.png "Model Dashboard")


Create Deployments to Triton Inference Server:

```bash
mlflow deployments create -t triton --flavor triton --name yolov6n -m models:/yolov6n/1

```
![Model loaded](imgs/model_loaded.png "Model loaded")
If you want to delete and update the model:
```bash
mlflow deployments delete -t triton --name yolov6n

mlflow deployments update -t triton --flavor triton --name yolov6n -m models:/yolov6n/2
```
## For infererencing we have 2 options:  

### 1 ) Perform inference with mlflow:  
  

```bash 
mlflow deployments predict -t triton --name yolov6n --input-path <path-to-the-examples-directory>/input.json --output-path output.json

#Example input json for yolov6n:
#img_ex is list of ndarray
{"inputs":[{"name":"images","datatype":"FP32","shape":[1, 3, 640, 640],"data":"example_image_array"}]}
```

### 2 ) Perform inference with triton http/gRPC client:  

```python
import tritonclient.http as httpclient

model_name = "yolov6n"
triton_client = httpclient.InferenceServerClient(url="0.0.0.0:8000")
triton_client.get_model_metadata(model_name)
```

{'name': 'yolov6n',
 'versions': ['1'],
 'platform': 'onnxruntime_onnx',
 'inputs': [{'name': 'images', 'datatype': 'FP32', 'shape': [1, 3, 640, 640]}],
 'outputs': [{'name': 'outputs', 'datatype': 'FP32', 'shape': [1, 8400, 85]}]}

```python
inputs = []
outputs = []
im = np.array(im, dtype=np.float32) # im is the image numpy array
inputs.append(httpclient.InferInput('images', [1,3,640,640], "FP32"))
outputs.append(httpclient.InferRequestedOutput("outputs"))
inputs[0].set_data_from_numpy(im)

results = triton_client.infer(model_name=model_name, inputs=inputs)
results.get_response()
```
{'model_name': 'yolov6n',
 'model_version': '1',
 'outputs': [{'name': 'outputs',
   'datatype': 'FP32',
   'shape': [1, 8400, 85],
   'parameters': {'binary_data_size': 2856000}}]}  

## <b><u>Conclusion</b></u>:  

 There are several advantages of using both mlflow and triton Inference server together:
   
We can utilize the inferecning power of Triton i.e.   

- Multi backend/arch supported inferencing.  
- Complete and optimized utilization of CPU/GPU resources.  
- Multi protocol support gRPC/Http  

And we have advantages of using mlflow as the frontend for Model Deployments and Model management,  
 
- Mlflow is lightweight and fully featured MLOps toolkit with tons of API's.  
- A centralized model store with UI to collaboratively manage the full lifecycle of ML Models.  
- Model Tracking throughout development and Deployment with experiments using an interactive UI.  


### <b>References</b>:  
https://github.com/mlflow/mlflow  
https://github.com/triton-inference-server  
https://github.com/triton-inference-server/client  
https://catalog.ngc.nvidia.com/orgs/nvidia/teams/morpheus/containers/mlflow-triton-plugin  
https://github.com/triton-inference-server/server/tree/r22.09/deploy/mlflow-triton-plugin
https://catalog.ngc.nvidia.com/orgs/nvidia/teams/morpheus/containers/mlflow-triton-plugin
https://github.com/nv-morpheus/Morpheus/tree/bc791eaec7ffa19db2fd292f8fb65a74473885a2/models/mlflow

