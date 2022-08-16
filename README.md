<h4>Redes Neurais Artificiais</h4>
<p align="justify">
<p align="justify">
O algoritmo de rede neural artificial é uma abstração matemática baseada no funcionamento da atividade neuronal do cérebro e tenda traduzir uma arquitetura analógica da biologia em uma arquitetura digital que possa ser computada de forma estritamente matemática. O primeiro algoritmo desse tipo a ser formulado foi o Perceptron que consiste em uma fórmula matemática abstraída das principais funções de um único neurônio, também conhecida como função de aproximação por tentar aproximar uma entrada de uma saída específica.<br>
</p>
<h5>Perceptron/Perceptron Simples</h5>
<p align="justify">
O Perceptron (ou Perceptron Simples) assim como todo algoritmo de aprendizado de máquina conta com duas fases (ou etapas) principais que são: treinamento e predição. Uma fase intermediária de teste também poderá ser utilizada caso o desenvolvedor deseje averiguar os erros e acertos antes de colocar o projeto em produção. <br>
Depois da implantação do projeto o treinamento não será mais necessário uma vez que os pesos responsáveis pela aproximação do resultado já terão sido computados e salvos em um modelo, bastando executar somente a fase de predição.<br>
O Perceptron também é conhecido como neurônio artificial por conter uma única estrutura neuronal sem camadas entre a entrada e a saída. Como todo algoritmo de aprendizado supervisionado o Perceptron exige a atribuição de uma lista com uma saída de exemplo para cada entrada do treinamento. Outros parâmetros que poderão ser definidos são o parâmetro “epochs” responsável por definir um limite no número de recálculos dos pesos, o parâmetro “learning_rate” que receberá um número real referente ao percentual de padrões que deverão ser assimilados, o parâmetro “bias” que receberá um número qualquer com o objetivo de aumentar ou diminuir os valores nas respostas, o parâmetro “activation_function” que irá definir a função que irá formatar a saída e o parâmetro “show_error” que quando definido como True exibe o progresso da fase de treinamento enquanto a mesma ocorre.<br>
</p>
<br>
<pre>
  <code>
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
  </code>
</pre>
<br>
<p align="justify">
Para o nosso exemplo estamos passando entradas e saídas de exemplo para o neurônio com o intuito de que ele aprenda os padrões do operador lógico OR. A maioria das linguagens de programação trabalha com operadores lógicos que são capazes de avaliar o estado de duas entradas a fim de se retornar um terceiro estado baseado nos dois anteriores. No caso do operador OR o retorno será 1 (True/verdadeiro) quando houver pelo menos uma entrada igual a 1 (True/verdadeiro), somente quando ambas as entradas forem iguais a 0 (False/falso) é que o retorno será igual a 0 (False/falso). Como o Perceptron só aceita números nós estamos representando o estado falso com 0 e o estado verdadeiro com 1.<br>
</div>
<br>
Resultado:
<pre>
  <code>
epoch...............................: 1 - loss: 2.81000000
epoch...............................: 2 - loss: 1.52945248
epoch...............................: 3 - loss: 1.32945247
epoch...............................: 4 - loss: 0.33986465
epoch...............................: 5 - loss: 0.13986464
[[0.0], [0.9262462174627673], [0.728838955793307], [0.9880608084158808]]
  </code>
</pre>
<br>
<p align="justify">
Observe que o erro/perda diminui drasticamente nas duas últimas iterações, isso ocorre por que o Neuraline é um algoritmo invasivo auto configurável, isso quer dizer que quando você não configura os parâmetros da melhor forma possível ele varre os dados antes de terminar o treinamento e reconfigura a arquitetura e os parâmetros do seu modelo para obter um resultado melhor quando ele percebe que a execução tem o risco de acabar antes que o valor seja aproximado. Isso acontece com todos os algoritmos do Neuraline por “debaixo dos panos” de forma totalmente automática sem que você perceba, diminuindo a necessidade de se ficar realizando vários testes até encontrar uma configuração que sirva.
</div>
<p align="justify">
Como o neurônio artificial é baseado na estrutura física do neurônio biológico que não computa de forma exata, o neurônio artificial também reproduzirá esse comportamento retornando resultados aproximados. Podemos considerar a resposta como sendo correta quando conseguimos chegar ao valor desejado arredondando as saídas da predição para a mesma quantidade de casas decimais das saídas do treinamento. Note que se arredondarmos os números de resposta para zero casas e os convertermos para inteiros como estavam no treinamento o resultado será igual ao que estávamos esperando (0 quando as duas entradas forem 0 e 1 quando pelo menos uma for 1).
</div>
<p align="justify">
Estamos utilizando a função de ativação da tangente hiperbólica (“tanh”) por que ela emite resultados entre -1 e 1 que são resultados mais próximos dos valores 0 e 1 que estamos querendo obter como resposta. Para este conjunto de dados é a que nos retorna os melhores resultados. Você deverá fazer testes com outras funções caso esteja utilizando dados diferentes até encontrar a função que emita os resultados que você espera. Geralmente usamos funções com intervalos maiores em dados com muitas possibilidades de resposta e funções com intervalos menores em dados com poucas possibilidades de resposta como este nosso exemplo que só possui 0 ou 1 como resultado.	
</div>
<p align="justify">
Se tentarmos aplicar o mesmo algoritmo na aprendizagem do operador lógico XOR não obteremos sucesso em todas as saídas. Isso ocorre por que o Perceptron Simples é limitado às camadas de entrada e saída, o que faz com que ele só consiga assimilar padrões lineares ou próximos de uma linearidade. Quando os pesos são recalculados de um backpropagation para outro temos pouca variação nos resultados já que os dados são transferidos diretamente da camada de entrada para a camada de saída fazendo com que a progressão tenda a resultados lineares. Como só temos 0 ou 1 como possibilidades de resposta a tendência será sempre uma evolução contínua de 0 para 1 sem voltar para 0 ou de 1 para 0 sem voltar para 1. Observe o exemplo a seguir:	
</div>
<br>
<pre>
  <code>
from Neuraline.ArtificialIntelligence.DeepLearning.feedforward import Perceptron
perceptron = Perceptron()
# valores de entrada e saída correspondentes ao operador lógico XOR
# não é possível que o Perceptron Simples aprenda a lógica do operador XOR
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
  </code>
</pre>
<br>
Resultado:
<pre>
  <code>
epoch...............................: 1 - loss: 2.81000000
epoch...............................: 2 - loss: 1.42654464
epoch...............................: 3 - loss: 1.22654463
epoch...............................: 4 - loss: 0.34949562
epoch...............................: 5 - loss: 0.14949561
[[0.0], [0.912435328986984], [0.7222990832371986], [0.98534295421847]]
  </code>
</pre>
<br>
<p align="justify">
Note na figura abaixo que para o operador OR só precisamos de uma única linha (linear) para separar as respostas do tipo 0 das respostas do tipo 1. Isso por que há uma progressão linear dos valores de saída, pois eles partem de 0 e uma vez que entram no estado 1 não retornam para o 0 mantendo a linearidade da progressão. Já no XOR os valores de saída partem de 0 para 1 e depois retornam para 0, ou seja, não seguem uma progressão linear e por isso são necessárias duas linhas para realizar a separação. Essa segunda linha extra é representada pela camada oculta da rede.  Quando colocamos uma camada extra entre a entrada e a saída os produtos das entradas pelos pesos sofrem uma nova sequência de multiplicações que retira a linearidade dos dados tornando possível o reconhecimento de padrões não lineares.
</p>
<br>
<div align="center"><img src="https://github.com/grupoquantum/redes_neurais_artificiais/blob/main/operadores_or_xor.png"></div>
<br>
<p align="justify">
Antes só precisávamos reconhecer dois padrões, um abaixo e outro acima da linha, agora precisamos reconhecer três, um na parte de baixo, um no meio e outro em cima. Para dois padrões ou menos as duas camadas default (entrada e saída) eram o suficiente, porém para três padrões precisaremos de três camadas, uma na entrada, uma no meio (oculta) e outra na saída. Quanto maior o número de camadas maior será a quantidade de padrões que poderão ser aprendidos, mas cuidado, ao exceder a quantidade de camadas para além do necessário você poderá estar assimilando padrões irrelevantes que irão influenciar o seu resultado sem que houvesse essa necessidade, em muitos casos resultando em respostas incorretas.
</p>
<h5>Multilayer Perceptron</h5>
<p align="justify">
Com o Multilayer Perceptron temos a capacidade de adicionar quantas camadas ocultas forem necessárias ao neurônio o transformando em uma rede neural artificial que conecta o neurônio formado pela entrada e saída a um ou mais neurônios formados pelas camadas ocultas. Como agora estaremos assimilando mais padrões do que antes teremos que aumentar o número de backpropagations no parâmetro “epochs” para dar mais tempo para os padrões serem assimilados e a função de ativação dessa vez deverá ser atribuída a camada oculta e não mais a camada de saída pelo método de treinamento, isso por que a formatação da camada oculta será propagada para a saída não havendo a necessidade de repetir a formatação. Dessa vez iremos utilizar a função de ativação linear, pois como a nossa rede perderá a linearidade com a camada oculta nós não queremos que a não linearidade seja muito exagerada já que temos apenas dois números (0 e 1). No parâmetro “hidden_nodes” da camada oculta definiremos o número de nós (pesos) que a nossa camada irá utilizar para multiplica-los pelos valores de entrada, geralmente definimos um nó a mais com relação à camada anterior, como a nossa camada de entrada recebe uma lista com dois elementos (dois nós) então na camada seguinte que é a oculta iremos definir três, mas isso não é uma regra, apenas um norte por onde iniciar os testes. Também iremos definir na nossa camada oculta o tipo de conexão densa com o parâmetro “dense” igual à True para aumentar a precisão no resultado. Camadas densas multiplicam cada nó por todos os nós da camada anterior e posterior, diferente da conexão esparsa que multiplica cada nó apenas uma única vez. Confira na imagem abaixo como ficaria a representação gráfica da nossa arquitetura:	
</p>
<br>
<div align="center"><img src="https://github.com/grupoquantum/redes_neurais_artificiais/blob/main/multilayer_perceptron.png"></div>
<br>
<p align="justify">
Temos duas entradas na camada de entrada com um único número de resposta na camada de saída e uma camada oculta linear densa com três nós entre elas. Nos nós da camada de entrada poderemos receber 0 com 0, 0 com 1, 1 com 0 ou 1 com 1 e na camada de saída teremos um único nó de resultado que poderá ser 0 ou 1.
</p>
<br>
<pre>
  <code>
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
  </code>
</pre>
<br>
Resultado:
<pre>
  <code>
epoch..............................: 01 - loss: 2.91000000
epoch..............................: 02 - loss: 2.81000000
epoch..............................: 03 - loss: 2.71000000
epoch..............................: 04 - loss: 2.61000000
epoch..............................: 05 - loss: 2.51000000
epoch..............................: 06 - loss: 1.05065041
epoch..............................: 07 - loss: 0.95065040
epoch..............................: 08 - loss: 0.85065039
epoch..............................: 09 - loss: 0.75065038
epoch..............................: 10 - loss: 0.13383204
[[0.0], [1.1299769555511598], [1.1217726158195234], [0.36598899333832596]]
  </code>
</pre>
<br>
<p align="justify">
É importante ressaltar que como os pesos são inicializados com valores aleatórios você poderá ter resultados diferentes de um treinamento para outro. Você poderá executar o treinamento mais de uma vez com os mesmos parâmetros antes de alterar a arquitetura para conferir se o resultado é capaz de se tornar mais satisfatório sem a necessidade de alteração das configurações. Se você estiver usando poucas épocas como no nosso exemplo, poderá ser sempre uma boa prática repetir a execução do treinamento em busca de resultados melhores.
</p>
<p align="justify">
Já que podemos repetir o treinamento por diversas vezes em busca de resultados melhores, como poderíamos salvar o melhor treinamento para utilizá-lo posteriormente em predições futuras? Com o Neuraline isso é extremamente simples, basta fazer uma chamada ao método “saveModel” contido em todos os algoritmos de aprendizagem da biblioteca após o treinamento para salvar o modelo com as configurações do treinamento atual eliminando a necessidade de novos treinamentos futuros. Se o método for chamado após a predição ele salvará além das configurações de treinamento as configurações da predição caso o método de predição do algoritmo em questão tenha parâmetros de configuração. Para carregar as configurações salvas basta fazer uma chamada ao método “loadModel” antes da predição.
</p>
<br>
<pre>
  <code>
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
  </code>
</pre>
<br>
Resultado:
<pre>
  <code>
epoch..............................: 01 - loss: 2.91000000
epoch..............................: 02 - loss: 2.81000000
epoch..............................: 03 - loss: 2.71000000
epoch..............................: 04 - loss: 2.61000000
epoch..............................: 05 - loss: 2.51000000
epoch..............................: 06 - loss: 1.00502068
epoch..............................: 07 - loss: 0.90502067
epoch..............................: 08 - loss: 0.80502066
epoch..............................: 09 - loss: 0.70502065
epoch..............................: 10 - loss: 0.18077929
  </code>
</pre>
<br>
<p align="justify">
Depois de salvar o último treinamento, iremos chamar o método “loadModel” para carregar as configurações do último treino executado. Dessa forma não há mais a necessidade de novos treinamentos, podemos simplesmente executar a predição de forma direta.
</p>
<br>
<pre>
  <code>
from Neuraline.ArtificialIntelligence.DeepLearning.feedforward import MultilayerPerceptron
multilayer_perceptron = MultilayerPerceptron()

''' carrega o treinamento do cache '''
multilayer_perceptron.loadModel()
new_inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
''' agora a execução será muito mais rápida por que o treinamento não é mais necessário '''
new_outputs = multilayer_perceptron.predict(inputs=new_inputs)
''' como não há mais recálculo dos pesos com o treinamento agora não haverá mais variação de resultado entre uma predição e outra '''
print(new_outputs)
  </code>
</pre>
<br>
Resultado:
<pre>
  <code>
[[0.0], [1.1570217710455415], [1.1631996102122475], [0.4750262774538163]]
  </code>
</pre>
<br>
<p align="justify">
Mas e se quisermos gerar um arquivo do nosso treinamento que possa ser copiado e compartilhado? Também é muito simples, basta definirmos um nome com um endereço optativo no parâmetro “url_path” do método “saveModel”. Se você definir só o nome sem um endereço de diretório o modelo será salvo na pasta local.
</p>
<br>
<pre>
  <code>
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
''' salva o treinamento no diretório local com o nome "model.ai" '''
multilayer_perceptron.saveModel(url_path='model')
'''
se desejar você poderá executar a predição após o salvamento para averiguar o tipo de resultado 
que a configuração que está sendo salva irá emitir em predições futuras
'''
new_inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
new_outputs = multilayer_perceptron.predict(inputs=new_inputs)
print(new_outputs)
  </code>
</pre>
<br>
Resultado:
<pre>
  <code>
epoch..............................: 01 - loss: 2.91000000
epoch..............................: 02 - loss: 2.81000000
epoch..............................: 03 - loss: 2.71000000
epoch..............................: 04 - loss: 2.61000000
epoch..............................: 05 - loss: 2.51000000
epoch..............................: 06 - loss: 1.02154166
epoch..............................: 07 - loss: 0.92154165
epoch..............................: 08 - loss: 0.82154164
epoch..............................: 09 - loss: 0.72154163
epoch..............................: 10 - loss: 0.16580793
[[0.0], [1.148804551472201], [1.154381145004061], [0.4039325691539646]]
  </code>
</pre>
<br>
<p align="justify">
Você poderá repetir a execução do algoritmo de salvamento quantas vezes quiser para tentar melhorar o resultado, quando o código de salvamento é executado a partir da segunda vez o treinamento anterior salvo com o mesmo nome será sobrescrito. Observe que agora toda vez que o algoritmo for executado com o carregamento deste arquivo o resultado será sempre o mesmo da predição do treinamento que foi salvo, com exceção de casos onde as entradas forem diferentes das que foram utilizadas no treino anterior.
</p>
<br>
<pre>
  <code>
from Neuraline.ArtificialIntelligence.DeepLearning.feedforward import MultilayerPerceptron
multilayer_perceptron = MultilayerPerceptron()

''' para carregar um modelo basta passar o caminho do arquivo no parâmetro url_path '''
multilayer_perceptron.loadModel(url_path='model')
new_inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
new_outputs = multilayer_perceptron.predict(inputs=new_inputs)
print(new_outputs)
  </code>
</pre>
<br>
Resultado:
<pre>
  <code>
[[0.0], [1.148804551472201], [1.154381145004061], [0.4039325691539646]]
  </code>
</pre>
<br>
<p align="justify">
Note que como usamos as mesmas entradas do treinamento que foi salvo o resultado foi exatamente o mesmo, porém agora com uma execução muito mais rápida do que teríamos se estivéssemos treinando o algoritmo a cada predição.
</p>
<h5>Radial Basis Function</h5>
<p align="justify">
Também podemos simular o operador lógico XOR com a Função de Base Radial que é uma estrutura Multilayer Perceptron de uma única camada oculta obrigatória que não precisará ser adicionada, os parâmetros dessa camada são definidos no próprio método de treinamento. Este é considerado um algoritmo de Aprendizado Profundo assim como qualquer rede neural com pelo menos uma camada oculta, diferente do Perceptron Simples que é somente um algoritmo de Aprendizado de Máquina comum por não adotar camadas extras na sua arquitetura. Chamamos de Aprendizado Profundo qualquer arquitetura de rede com uma ou mais camadas ocultas.
</p>
<br>
<pre>
  <code>
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
  </code>
</pre>
<br>
Resultado:
<pre>
  <code>
epoch..............................: 01 - loss: 2.91000000
epoch..............................: 02 - loss: 2.81000000
epoch..............................: 03 - loss: 2.71000000
epoch..............................: 04 - loss: 2.61000000
epoch..............................: 05 - loss: 2.51000000
epoch..............................: 06 - loss: 0.96836999
epoch..............................: 07 - loss: 0.86836998
epoch..............................: 08 - loss: 0.76836997
epoch..............................: 09 - loss: 0.66836996
epoch..............................: 10 - loss: 0.13483660
[[0.0], [1.11992776638833], [1.1323194978987858], [0.39979931587721085]]
  </code>
</pre>
<br>
<h5>Deep Feedforward</h5>
<p align="justify">
Outra arquitetura de rede que poderá ser utilizada é a Deep Feedforward que possui uma arquitetura equivalente ao Multilayer Perceptron. Como toda arquitetura do tipo Feedforward ela usa pesos inicialmente aleatórios na primeira época e os ajusta a cada nova iteração. Por isso você poderá executar o treinamento repetidas vezes enquanto salva os modelos dos treinos executados até que o resultado seja satisfatório e o último arquivo salvo não precise mais ser sobrescrito. Somente depois de executar o treinamento repetidas vezes é que você deverá testar novas combinações de parâmetros caso o objetivo não seja alcançado. Existem alguns modelos de redes neurais do Google por exemplo que cada execução do treinamento leva dias ou até meses para ser concluída devido a quantidade e complexidade dos dados. A vantagem das redes neurais é que esse processo de treinamento só é feito uma única vez e a predição poderá ser executada de forma relativamente mais rápida simplesmente carregando o último modelo salvo.
</p>
<br>
<pre>
  <code>
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
  </code>
</pre>
<br>
Resultado:
<pre>
  <code>
epoch..............................: 01 - loss: 2.94333333
epoch..............................: 02 - loss: 2.87666667
epoch..............................: 03 - loss: 2.81000000
epoch..............................: 04 - loss: 2.74333333
epoch..............................: 05 - loss: 1.24021188
epoch..............................: 06 - loss: 1.17354521
epoch..............................: 07 - loss: 1.10687853
epoch..............................: 08 - loss: 1.04021185
epoch..............................: 09 - loss: 0.97354518
epoch..............................: 10 - loss: 0.90687850
epoch..............................: 11 - loss: 0.84021182
epoch..............................: 12 - loss: 0.77354515
epoch..............................: 13 - loss: 0.70687847
epoch..............................: 14 - loss: 0.64021179
epoch..............................: 15 - loss: 0.13484568
[[0.0], [1.1287355037864766], [1.125865335288039], [0.3383126255924018]]
  </code>
</pre>
<br>
<pre>
  <code>
from Neuraline.ArtificialIntelligence.DeepLearning.feedforward import DeepFeedForward
deep_feedforward = DeepFeedForward()
''' com um modelo pré-treinado não há mais a necessidade de se treinar o modelo '''
deep_feedforward.loadModel('modelo_deep_feedforward') # nome do arquivo modelo
new_inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
new_outputs = deep_feedforward.predict(inputs=new_inputs)
''' ao carregar um arquivo modelo o resultado será sempre equivalente ao do último treinamento salvo '''
print(new_outputs)
  </code>
</pre>
<br>
Resultado:
<pre>
  <code>
[[0.0], [1.1287355037864766], [1.125865335288039], [0.3383126255924018]]
  </code>
</pre>
<br>
<h5>Neural Network</h5>
<p align="justify">
Existe um outro algoritmo de Rede Neural Artificial que podemos utilizar que é o algoritmo Neural Network. Esse algoritmo interfere nos seus parâmetros de configuração com o objetivo de acertá-los de maneira muito mais intensa do que os demais algoritmos de redes neurais que vimos até agora. A vantagem é que ele exige menos tempo de configuração tornando-se mais produtivo, porém isso faz com que ele seja menos flexível e personalizável do que os outros. Para usuários iniciantes ele é bem mais prático de se programar devido a sua inteligência artificial interna de autoconfiguração. Em alguns casos ele poderá ser consideravelmente mais rápido na fase de treinamento por permitir que um número muito menor de épocas seja utilizado para emitir o mesmo resultado que conseguiríamos em outras arquiteturas com muito mais épocas. Para o nosso exemplo iremos fazer com que a rede neural aprenda a lógica do operador NOT que simplesmente inverte a polaridade das entradas transformando 0 (False/falso) em 1 (True/verdadeiro) e 1 (True/verdadeiro) em 0 (False/falso).
</p>
<br>
<pre>
  <code>
from Neuraline.ArtificialIntelligence.DeepLearning.neural_network import NeuralNetwork
neural_network = NeuralNetwork()
# valores de entrada e saída correspondentes ao operador lógico NOT
inputs, outputs = [[0], [1]], [[1], [0]]
neural_network.fit(
	inputs=inputs, 
	outputs=outputs, 
	epochs=5,
	learning_rate=1,
	bias=0,
	activation_function='binary_step', # a função binary step retornará 0 para valores menores que 0.5 e 1 para valores maiores ou iguais a 0.5
	show_error=True
)
new_inputs = [[1], [0]]
new_outputs = neural_network.predict(inputs=new_inputs)
print(new_outputs) # com a função binary step conseguimos resultados exatados quando as saídas possíveis forem 0 e 1
  </code>
</pre>
<br>
Resultado:
<pre>
  <code>
epoch...............................: 1 - loss: 0.80000000
epoch...............................: 2 - loss: 0.60000000
epoch...............................: 3 - loss: 0.40000000
epoch...............................: 4 - loss: 0.20000000
epoch...............................: 5 - loss: 0.00000000
[[0], [1]]
  </code>
</pre>
<br>
<p align="justify">
Note que como esse algoritmo é mais invasivo nas autoconfigurações o erro tende a zero de forma mais intensa na última época para forçar ao máximo o resultado correto.
</p>
<p align="justify">
Confira agora como ficaria o mesmo algoritmo para a assimilação da aprendizagem do operador lógico AND que retornará 1 (True/verdadeiro) somente quando ambas as alternativas forem iguais a 1 (True/verdadeiro).
</p>
<br>
<pre>
  <code>
from Neuraline.ArtificialIntelligence.DeepLearning.neural_network import NeuralNetwork
neural_network = NeuralNetwork()
# valores de entrada e saída correspondentes ao operador lógico AND
inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
outputs = [[0], [0], [0], [1]]
neural_network.fit(
	inputs=inputs, 
	outputs=outputs, 
	epochs=5,
	learning_rate=1,
	bias=0,
	activation_function='binary_step',
	show_error=True
)
new_inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
new_outputs = neural_network.predict(inputs=new_inputs)
print(new_outputs)
  </code>
</pre>
<br>
Resultado:
<pre>
  <code>
epoch...............................: 1 - loss: 0.80000000
epoch...............................: 2 - loss: 0.60000000
epoch...............................: 3 - loss: 0.40000000
epoch...............................: 4 - loss: 0.20000000
epoch...............................: 5 - loss: 0.00000000
[[0], [0], [0], [1]]
  </code>
</pre>
<br>
<p align="justify">
Agora observe como ficaria a aplicação do código para a assimilação da aprendizagem do operador lógico OR que retornará 1 (True/verdadeiro) sempre que houver pelo menos uma entrada igual a 1 (True/verdadeiro).
</p>
<br>
<pre>
  <code>
from Neuraline.ArtificialIntelligence.DeepLearning.neural_network import NeuralNetwork
neural_network = NeuralNetwork()
# valores de entrada e saída correspondentes ao operador lógico OR
inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
outputs = [[0], [1], [1], [1]]
neural_network.fit(
	inputs=inputs, 
	outputs=outputs, 
	epochs=5,
	learning_rate=1,
	bias=0,
	activation_function='binary_step',
	show_error=True
)
new_inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
new_outputs = neural_network.predict(inputs=new_inputs)
print(new_outputs)
  </code>
</pre>
<br>
Resultado:
<pre>
  <code>
epoch...............................: 1 - loss: 0.80000000
epoch...............................: 2 - loss: 0.60000000
epoch...............................: 3 - loss: 0.40000000
epoch...............................: 4 - loss: 0.20000000
epoch...............................: 5 - loss: 0.00000000
[[0], [1], [1], [1]]
  </code>
</pre>
<br>
<p align="justify">
Lembre-se de que padrões que podem ser separados por uma única linha na tabela de dados não necessitam de camada oculta para assimilar todos os padrões envolvidos, mas padrões como os do operador lógico XOR que não podem ser classificados de forma linear precisam de pelo menos uma camada oculta para assimilar todos os padrões. Confira na figura a seguir a comparação dos operadores distribuídos de forma tabular em uma tabela de dados para relembrarmos o conceito, a coluna F representa os Fields/Campos de saída:
</p>
<br>
<div align="justify"><img src="https://github.com/grupoquantum/redes_neurais_artificiais/blob/main/operadores_and_or_xor.jpg"></div>
<br>
<p align="justify">
É possível notar na figura acima que enquanto os operadores AND e OR podem ser separados por uma única linha que divide a tabela em duas regiões diferentes, sendo a região superior referente aos outputs do tipo 0 e a região inferior referente aos outputs do tipo 1, o mesmo não pode ser dito do operador XOR que necessita de duas linhas para separar os outputs do tipo 0 dos outputs do tipo 1. Observe na figura abaixo que a dificuldade também se repete ao tentarmos fazer a separação do XOR com uma única linha no plano cartesiano.
</p>
<br>
<div align="justify"><img src="https://github.com/grupoquantum/redes_neurais_artificiais/blob/main/operadores_and_or_xor_no_plano_cartesiano.png"></div>
<br>
<p align="justify">
Na imagem acima temos um padrão diagonal de descendência linear para os operadores AND e OR, mas não é possível aplicar a classificação na distribuição do XOR da mesma forma que fizemos com os outros operadores. Logo deveríamos acrescentar uma camada oculta no nosso algoritmo correto? Errado. O algoritmo Neural Network é tão poderoso que é capaz de perceber essa não linearidade exigida nos dados e acrescentar por conta própria uma camada oculta em tempo de execução sem que você sequer dê conta disso. Confira como é muito mais simples fazer isso com a Neural Network observando o código do algoritmo a seguir:
</p>
<br>
<pre>
  <code>
from Neuraline.ArtificialIntelligence.DeepLearning.neural_network import NeuralNetwork
neural_network = NeuralNetwork()
# valores de entrada e saída correspondentes ao operador lógico XOR
inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
outputs = [[0], [1], [1], [0]]
neural_network.fit(
	inputs=inputs, 
	outputs=outputs, 
	epochs=5,
	learning_rate=1,
	bias=0,
	activation_function='binary_step',
	show_error=True
)
new_inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
new_outputs = neural_network.predict(inputs=new_inputs)
print(new_outputs)
  </code>
</pre>
<br>
Resultado:
<pre>
  <code>
epoch...............................: 1 - loss: 0.80000000
epoch...............................: 2 - loss: 0.60000000
epoch...............................: 3 - loss: 0.40000000
epoch...............................: 4 - loss: 0.20000000
epoch...............................: 5 - loss: 0.00000000
[[0], [1], [1], [0]]
  </code>
</pre>
<br>
<p align="justify">
Aparentemente o algoritmo parece não ter camadas ocultas, mas ele tem, a camada oculta devidamente configurada foi acrescentada “por debaixo dos panos” de forma totalmente automatizada e o resultado desejado pôde ser obtido de forma incrivelmente exata.
</p>
<p align="justify">
Mas e se quiséssemos adicionar camadas ocultas mesmo assim? Neste caso bastaria fazer uma chamada ao método “addHiddenLayer” da mesma forma que fizemos com as outras arquiteturas. Porém aqui não temos a opção de habilitar ou desabilitar a densidade das conexões neuronais por que isso é feito de forma automática pelo algoritmo. Em contrapartida poderemos definir uma função de ativação para o treinamento diferente da utilizada na camada oculta já que o parâmetro da função de ativação existe tanto no método “addHiddenLayer” quanto no método “fit”. Porém é importante ressaltar que ao adicionarmos as camadas manualmente estaremos eliminando a configuração automática de camadas que poderá influenciar diretamente na taxa de erro, então antes de resolver alterar as configurações manualmente é bom que você saiba o que está fazendo.
</p>
<br>
<pre>
  <code>
from Neuraline.ArtificialIntelligence.DeepLearning.neural_network import NeuralNetwork
neural_network = NeuralNetwork()
# valores de entrada e saída correspondentes ao operador lógico XOR
inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
outputs = [[0], [1], [1], [0]]
# acréscimo da camada oculta de forma manual
neural_network.addHiddenLayer(hidden_nodes=3, activation_function='linear')
neural_network.fit(
	inputs=inputs, 
	outputs=outputs, 
	epochs=5,
	learning_rate=1,
	bias=0,
	activation_function='binary_step',
	show_error=True
)
new_inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
new_outputs = neural_network.predict(inputs=new_inputs)
print(new_outputs)
  </code>
</pre>
<br>
Resultado:
<pre>
  <code>
epoch...............................: 1 - loss: 9.00000000
epoch...............................: 2 - loss: 1.30400000
epoch...............................: 3 - loss: 1.30399999
epoch...............................: 4 - loss: 0.10000000
epoch...............................: 5 - loss: 0.00040000
[[0], [1], [1], [0]]
  </code>
</pre>
<br>
<p align="justify">
Observe que agora que configuramos a camada manualmente o treinamento já teve um pouco mais de dificuldade em acertar os demais parâmetros de forma automática por que ele foi obrigado a manter o algoritmo com uma única camada que foi configurada por nós e isso influenciou diretamente a taxa de erro que agora não termina mais no zero absoluto.
</p>
<p align="justify">
Se quiséssemos adicionar múltiplas camadas ocultas com configurações diferentes o procedimento seria o mesmo independentemente da quantidade de camadas. No código a seguir estamos aplicando uma função de ativação da tangente hiperbólica na primeira camada oculta com 3 nós, uma função de ativação sigmoide para diminuir o intervalor anterior na segunda com 4 nós e uma função de ativação linear na última para deixar os resultados livres sem limitações com 3 nós. Essa técnica onde colocamos menos nós nas extremidades e mais no meio é conhecida como triangulação e ajuda a aumentar a variabilidade nos resultados, porém aqui usamos somente para exemplificar.
</p>
<br>
<pre>
  <code>
from Neuraline.ArtificialIntelligence.DeepLearning.neural_network import NeuralNetwork
neural_network = NeuralNetwork()
# valores de entrada e saída correspondentes ao operador lógico XOR
inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
outputs = [[0], [1], [1], [0]]
# acréscimo de múltiplas camadas ocultas de forma manual
neural_network.addHiddenLayer(hidden_nodes=3, activation_function=   'tanh')
neural_network.addHiddenLayer(hidden_nodes=4, activation_function='sigmoid')
neural_network.addHiddenLayer(hidden_nodes=3, activation_function= 'linear')
neural_network.fit(
	inputs=inputs, 
	outputs=outputs, 
	epochs=5,
	learning_rate=1,
	bias=0,
	activation_function='binary_step',
	show_error=True
)
new_inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
new_outputs = neural_network.predict(inputs=new_inputs)
print(new_outputs)
  </code>
</pre>
<br>
Resultado:
<pre>
  <code>
epoch...............................: 1 - loss: 8.99999999
epoch...............................: 2 - loss: 1.26597096
epoch...............................: 3 - loss: 0.20040000
epoch...............................: 4 - loss: 0.10065000
epoch...............................: 5 - loss: 0.00105000
[[0], [1], [1], [0]]
  </code>
</pre>
<br>
<p align="justify">
Note que agora obtivemos uma taxa de erro levemente maior, apesar de não ser nada expressivo que atrapalhe o resultado já é um indício de que a nossa configuração manual aumentou e devemos tomar mais cuidado ao configurar o restante dos parâmetros.
</p>
<p align="justify">
Uma das principais vantagens das redes neurais com relação a outros algoritmos de aprendizado de máquina é que elas podem ser utilizadas tanto em casos de classificação quanto em casos de regressão com respostas adaptativas. Diferente dos algoritmos de aprendizado de máquina que são especialistas e costumam se sair muito mal quando aplicados em predições que fogem a sua especialidade classificativa ou regressiva, as redes neurais são generalistas e se saem muito bem em ambos os casos quando configuradas corretamente. Por exemplo, o KNN é excelente para classificar dados, mas é extremamente limitado para regredir, já a Regressão Linear é excelente para regredir e totalmente desaconselhável para classificar. Mas as Redes Neurais Artificiais conseguem performar com resultados satisfatórios nas duas situações.
</p>
<p align="justify">
Confira o caso a seguir onde estamos passando dados regressivos para a nossa rede descobrir que para chegar na saída desejada ela deverá somar os dois elementos da entrada. Iremos utilizar um padrão linear simples de progressão numérica.
</p>
<br>
<pre>
  <code>
from Neuraline.ArtificialIntelligence.DeepLearning.neural_network import NeuralNetwork
neural_network = NeuralNetwork()
''' padrão linear onde cada saída representa a soma das entradas '''
inputs = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16], [17, 18], [19, 20]]
outputs = [[3], [7], [11], [15], [19], [23], [27], [31], [35], [39]]
''' camada oculta para o reconhecimento do padrão linear '''
neural_network.addHiddenLayer(hidden_nodes=3, activation_function= 'linear')
neural_network.fit(
	inputs=inputs, 
	outputs=outputs, 
	epochs=5,
	learning_rate=1,
	bias=.5, # viés com acréscimo de 0.5 para subir os valores e torna-los mais próximos do resultados esperados
	activation_function='nonlinear', # a função não linear poderá ser aplicada tanto em padrões lineares quanto não lineares
	show_error=True
)
neural_network.saveModel('neural_network_regressao_linear') # salvamento do modelo
'''
enquanto a função linear força resultados com o mesmo intervalo de diferneça, 
a função nonlinear (não linear) retorna os resultados sem qualquer tipo de padronição
e por isso poderá servir tanto para casos lineares quanto para casos não lineares 
retornando os dados com os resultados reais da aproximação
'''
new_inputs = [[2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13], [14, 15], [16, 17], [18, 19], [20, 21]]
new_outputs = neural_network.predict(inputs=new_inputs)
print(new_outputs)
  </code>
</pre>
<br>
Resultado:
<pre>
  <code>
epoch...............................: 1 - loss: 1.24114409
epoch...............................: 2 - loss: 0.59890000
epoch...............................: 3 - loss: 0.40170000
epoch...............................: 4 - loss: 0.20570000
epoch...............................: 5 - loss: 0.00970000
[[5.4605], [9.4253], [13.3869], [17.3453], [21.3005], [25.252499999999998], [29.201300000000003], [33.146899999999995], [37.0893], [41.0285]]
  </code>
</pre>
<br>
<p align="justify">
Note que com poucas épocas conseguimos chegar a resultados satisfatórios com uma baixíssima taxa de erro. Para conferir os resultados basta somar as entradas da predição.
</p>
<br>
<pre>
  <code>
from Neuraline.ArtificialIntelligence.DeepLearning.neural_network import NeuralNetwork
neural_network = NeuralNetwork()

neural_network.loadModel('neural_network_regressao_linear') # carregamento do modelo
new_inputs = [[2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13], [14, 15], [16, 17], [18, 19], [20, 21]]
new_outputs = neural_network.predict(inputs=new_inputs)
print(new_outputs)
  </code>
</pre>
<br>
Resultado:
<pre>
  <code>
[[5.4605], [9.4253], [13.3869], [17.3453], [21.3005], [25.252499999999998], [29.201300000000003], [33.146899999999995], [37.0893], [41.0285]]
  </code>
</pre>
<br>
<p align="justify">
Com o carregamento do modelo salvo na etapa anterior obteremos exatamente os mesmos resultados para as mesmas entradas. Os resultados só serão diferentes se as entradas da predição atual forem diferentes dos resultados da predição do salvamento.
</p>
</p>
