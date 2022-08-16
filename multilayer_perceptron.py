from Neuraline.ArtificialIntelligence.DeepLearning.feedforward import MultilayerPerceptron
multilayer_perceptron = MultilayerPerceptron()
# valores de entrada e saída correspondentes ao operador lógico XOR
# com o Multilayer Perceptron é possível acrescentar camadas ocultas capazes de aprender padrões não lineares
inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
outputs = [[0], [1], [1], [0]]
# acréscimo de uma camada oculta intermediária entre a entrada e a saída
multilayer_perceptron.addHiddenLayer(hidden_nodes=3, dense=True, activation_function='linear')
multilayer_perceptron.fit(
	inputs=inputs,
	outputs=outputs,
	epochs=10,
	learning_rate=1,
	bias=0,
	show_error=True
)
new_inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
new_outputs = multilayer_perceptron.predict(inputs=new_inputs)
print(new_outputs)