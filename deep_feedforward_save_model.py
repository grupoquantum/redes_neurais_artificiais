from Neuraline.ArtificialIntelligence.DeepLearning.feedforward import DeepFeedForward
deep_feedforward = DeepFeedForward()
''' a Deep Feedforward é uma arquitetura de rede equivalente ao Multilayer Perceptron e poderá ser construída da mesma forma '''
inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
outputs = [[0], [1], [1], [0]]

deep_feedforward.addHiddenLayer(hidden_nodes=3, dense=True, activation_function='linear')
''' execute o treinamento quantas vezes for necessário até que os resultados obtidos estejam corretos '''
deep_feedforward.fit(
	inputs=inputs,
	outputs=outputs,
	epochs=15, # dessa vez foram utilizadas 15 épocas, com o salvamento do modelo a quantidade de épocas poderá ser maior por que o treinamento não precisará se repetir com a predição
	learning_rate=1,
	bias=0,
	show_error=True
)
deep_feedforward.saveModel('modelo_deep_feedforward')
new_inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
new_outputs = deep_feedforward.predict(inputs=new_inputs)
''' lembre-se de que as suas respostas poderão ser diferentes por causa dos pesos aleatórios '''
print(new_outputs)