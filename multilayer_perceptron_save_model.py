from Neuraline.ArtificialIntelligence.DeepLearning.feedforward import MultilayerPerceptron
multilayer_perceptron = MultilayerPerceptron()
inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
outputs = [[0], [1], [1], [0]]
multilayer_perceptron.addHiddenLayer(hidden_nodes=3, dense=True, activation_function='linear')
multilayer_perceptron.fit(
	inputs=inputs,
	outputs=outputs,
	epochs=10,
	learning_rate=1,
	bias=0,
	show_error=True
)
# salva o treinamento em cache
multilayer_perceptron.saveModel()