from Neuraline.ArtificialIntelligence.DeepLearning.feedforward import MultilayerPerceptron
multilayer_perceptron = MultilayerPerceptron()

''' para carregar um modelo basta passar o caminho do arquivo no parâmetro url_path '''
multilayer_perceptron.loadModel(url_path='model')
new_inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
new_outputs = multilayer_perceptron.predict(inputs=new_inputs)
print(new_outputs)