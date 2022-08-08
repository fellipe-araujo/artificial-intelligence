import sys
import json

def create_model_settings_json(filename, content):
	f = open(filename, 'w')
	f.write(content)
	f.close()

if __name__ == "__main__":
  print(len(sys.argv))
  if len(sys.argv) < 2:
    print('Use: python3 inference_request.py <model_id>')
    exit(0)

  # model id
  # 037d56ab851c4ef985b3b6a9790b0203
  content = {
    "name": "wine-classifier",
    "implementation": "mlserver_mlflow.MLflowRuntime",
    "parameters": {
      "uri": "/home/fellipe/Documentos/AI.Lab/artificial-intelligence/segmentation_pages/MLflow/mlruns/0/" + str(sys.argv[1]) + "/artifacts/model"
    }
  }

  create_model_settings_json('model-settings.json', json.dumps(content))
