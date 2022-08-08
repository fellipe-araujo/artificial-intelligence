import requests

def send_inference_request():
	y_test = [
		"alcohol",
		"chlorides",
		"citric acid",
		"density",
		"fixed acidity",
		"free sulfur dioxide",
		"pH",
		"residual sugar",
		"sulphates",
		"total sulfur dioxide",
		"volatile acidity",
	]

	x_test = [7.4,0.7,0,1.9,0.076,11,34,0.9978,3.51,0.56,9.4]

	inference_request = {
		"columns": y_test,
		"data": [x_test],
	}
	# print (inference_request)

	endpoint = "http://localhost:8080/invocations"
	response = requests.post(endpoint, json=inference_request)

	# endpoint = "http://localhost:8080/v2/models/wine-classifier/infer"
	# response = requests.get(endpoint)

	return response.json()

if __name__ == "__main__":
	response = send_inference_request()

	print(response)