# FORMATO DA REGRA

Cada linha dos arquivos representa uma regra que está estruturada da seguinte forma:

> *[100, 'OVERLAP', 100.0, "self.f.event_pos(tokenE) == 'NOUN' and self.f.event_timex3_no_between_order(tokenE, tokenT) == True", 'RIPPER', 0.87, 113, 130]*

> *[código da regra, 'TIPO DA RELAÇÃO TEMPORAL', ordem, "expressão lógica que representa a regra", 'algoritmo de origem', acurácia, total de acertos, número de vezes que foi acionada]*

A expressão lógica que representa a regra é composta por conjunções de condições. Cada condição é constituída por uma *`feature`* de nosso [conjunto de *features*](../conjunto_de_features.md), um `operador`, que pode ser de igualdade ou desigualdade, e o `valor` da *feature*.

A expressão `self.f.` indica para sistema a classe onde está implementada a *`features`*.

Os argumentos `tokenE` e `tokenT`, representam o evento e a expressão temporal, respectivamente.

A `acurácia` é calculada como a quantidade de acertos dividido pela quantidade de instâncias classificadas pela regra:

$$
acurácia = \frac{acertos}{instâncias \ \ classificadas}
$$