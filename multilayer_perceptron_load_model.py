from Neuraline.ArtificialIntelligence.DeepLearning.feedforward import MultilayerPerceptron
multilayer_perceptron = MultilayerPerceptron()

''' carrega o treinamento do cache '''
multilayer_perceptron.loadModel()
new_inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
''' agora a execução será muito mais rápida por que o treinamento não é mais necessário '''
new_outputs = multilayer_perceptron.predict(inputs=new_inputs)
''' como não há mais recálculo dos pesos com o treinamento agora não haverá mais variação de resultado entre uma predição e outra '''
print(new_outputs)