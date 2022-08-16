from Neuraline.ArtificialIntelligence.DeepLearning.feedforward import DeepFeedForward
deep_feedforward = DeepFeedForward()
''' com um modelo pré-treinado não há mais a necessidade de se treinar o modelo '''
deep_feedforward.loadModel('modelo_deep_feedforward') # nome do arquivo modelo
new_inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
new_outputs = deep_feedforward.predict(inputs=new_inputs)
''' ao carregar um arquivo modelo o resultado será sempre equivalente ao do último treinamento salvo '''
print(new_outputs)