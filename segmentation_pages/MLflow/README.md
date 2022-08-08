## Train, Serve, and Score a Linear Regression Model (MLflow)

Este tutorial usa um conjunto de dados para prever a qualidade do vinho com base em características quantitativas como “acidez fixa”, “pH”, “açúcar residual” do vinho e assim por diante. Na execução deste tutorial, será feito:

* Treinamento de um modelo de regressão linear;
* Empacotamento do código que treina o modelo em um formato de modelo reutilizável e reproduzível

## Execução

Para executar o tutorial de exemplo, basta rodar o seguinte comando:

```bash
python3 sklearn_elasticnet_wine/train.py
```

É possível passar outros valores para os parâmetros alpha e l1_ratio:

```bash
python3 sklearn_elasticnet_wine/train.py <alpha> <l1_ratio>
```

**Obs: sempre que uma execução é feita, o MLflow registra as informações sobre as execuções do experimentos no diretório `/mlruns`**.

## Comparar os Modelos

Para a comparação dos modelos produzidos, o MLflow disponibiliza uma interface de usuário. Estando no mesmo diretório que contém o `/mlruns`, execute o seguinte comando:

```bash
mlflow ui
```

## Predição de novos dados (MLServer - Seldon | MLflow) - REST API

Para obter a predição de novos dados utilizando o MLServer - Seldon, será mostrado abaixo as seguintes etapas:

1) Será necessário obter o **model_id** do modelo que se quer utilizar. Para encontrá-lo, pode-se procurar tanto no diretório `mlruns/0/<model_id>/` quanto na *user interface* quando executamos `mlflow ui`;
2) No arquivo `config_model_settings.py` na **linha 21**, defina o caminho para o modelo desejado;
3) É preciso criar um arquivo de configuração a respeito do modelo que se quer utilizar e, em seguida, executar o *mlserver* para deixá-lo pronto para receber inferências. Para isto, execute o código abaixo:

```bash
python3 config_model_settings.py <model_id> & mlserver start .
```

4) Em um segundo terminal, execute o arquivo responsável por fazer a inferência de novos dados com o seguinte comando:

```bash
python3 inference_request.py
```