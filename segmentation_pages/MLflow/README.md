## Train, Serve, and Score a Linear Regression Model (MLflow)

Este tutorial usa um conjunto de dados para prever a qualidade do vinho com base em características quantitativas como “acidez fixa”, “pH”, “açúcar residual” do vinho e assim por diante. Na execução deste tutorial, será feito:

* Treinamento de um modelo de regressão linear;
* Empacotamento do código que treina o modelo em um formato de modelo reutilizável e reproduzível

## Execução

Para executar o tutorial de exemplo, basta rodar o seguinte comando:

```py
python3 sklearn_elasticnet_wine/train.py
```

É possível passar outros valores para os parâmetros alpha e l1_ratio:

```py
python3 sklearn_elasticnet_wine/train.py <alpha> <l1_ratio>
```

**Obs: sempre que uma execução é feita, o MLflow registra as informações sobre as execuções do experimentos no diretório `/mlruns`**.

## Comparar os Modelos

Para a comparação dos modelos produzidos, o MLflow disponibiliza uma interface de usuário. Estando no mesmo diretório que contém o `/mlruns`, execute o seguinte comando:

```py
mlflow ui
```