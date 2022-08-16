from Neuraline.ArtificialIntelligence.DeepLearning.feedforward import Perceptron
perceptron = Perceptron()
# valores de entrada e saída correspondentes ao operador lógico XOR
# não é possível que o Perceptron simples aprenda a lógica do operador XOR
inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
outputs = [[0], [1], [1], [0]]
perceptron.fit(
	inputs=inputs,
	outputs=outputs,
	epochs=5,
	learning_rate=1,
	bias=0,
	activation_function='tanh',
	show_error=True
)
new_inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
new_outputs = perceptron.predict(inputs=new_inputs)
print(new_outputs) # os resultados serão semelhantes aos do operador OR que é o operador mais próximo do XOR