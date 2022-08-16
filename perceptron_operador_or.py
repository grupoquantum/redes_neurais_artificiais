'''
importação da classe Perceptron contida no algoritmo feedforward
do submódulo de Aprendizado Profundo no módulo de Inteligência Artificial
'''
from Neuraline.ArtificialIntelligence.DeepLearning.feedforward import Perceptron
perceptron = Perceptron() # instanciação da classe Perceptron na variável objeto perceptron
# valores de entrada e saída correspondentes ao operador lógico OR
inputs = [[0, 0], [0, 1], [1, 0], [1, 1]] # lista com as alternativas de entrada
outputs = [[0], [1], [1], [1]] # lista com as saídas desejadas para cada tipo de entrada
# etapa de treinamento
perceptron.fit(
	inputs=inputs, # atribuição da lista de entradas
	outputs=outputs, # atribuição da lista de saídas
	epochs=5, # número de épocas para o limite de repetições do backpropagation
	learning_rate=1, # taxa de aprendizagem para a assimilação dos padrões
	bias=0, # valor de viés que forçará os resultados para cima ou para baixo
	activation_function='tanh', # função de ativação que formatará os resultados da camada de saída
	show_error=True # exibição do progresso do treinamento com a exibição da taxa de erro/perda
)
# lista com as entradas para as quais queremos as respostas com os mesmos padrões de saída do treinamento
new_inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
# etapa de predição
new_outputs = perceptron.predict(inputs=new_inputs) # as saídas para as entradas em new_inputs serão armazenadas em new_outputs
print(new_outputs) # exibição dos resultados preditivos