from Neuraline.ArtificialIntelligence.DeepLearning.feedforward import RadialBasisFunction
radial_basis_function = RadialBasisFunction()

inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
outputs = [[0], [1], [1], [0]]
'''
a função de base radial usa uma arquitetura multilayer perceptron com uma camada oculta fixa que sempre irá existir
ela facilita a construção do código em casos onde sabemos que teremos que utilizar uma única camada extra
'''
radial_basis_function.fit(
	inputs=inputs, # padrão de entrada
	outputs=outputs, # padrão de saída com as respostas de exemplo
	epochs=10, # 10 backpropagations serão executados
	learning_rate=1, # 100% dos padrões serão assimilados
	bias=-.25, # retira 25% dos valores resultantes para tornar as respostas menores
	show_error=True, # exibe o progresso do treinamento com a evolução da aprendizagem
	hidden_nodes=3, # número de neurônios/nós da camada oculta fixa
	dense=True, # conexão densa habilitada para a camada oculta fixa
	activation_function='linear' # função linear na formatação dos dados da camada oculta fixa
)
new_inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
new_outputs = radial_basis_function.predict(inputs=new_inputs)
print(new_outputs)