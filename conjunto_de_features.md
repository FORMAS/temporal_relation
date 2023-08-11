# CONJUNTO DE <i>FEATURES</i> 

Apresentaremos o nosso conjunto de *features* organizadas por tipo de informações linguísticas utilizado no desenvolvimento das regras aplicadas nesse trabalho. 

## Anotação TimeML
  - `event-class:` classe do evento anotada no <i>corpus</i>
  - `event-closest-to-event-class:` classe anotada do evento mais próximo do evento que está na relação temporal sob classificação
  - `event-closest-to-event-equal-class:` verifica se o valor da classe do evento da relação temporal sob classificação é igual à classe do evento mais próximo a ele no texto
  - `event-intervening-preceding-class:` classe do evento que é mencionado entre a expressão temporal e o evento, nesta ordem textual, da relação temporal sob classificação e está mais próximo da expressão temporal.  
     Por exemplo,  `TIMEX --- event2.class ------------ EVENT`
  - `event-polarity:` polaridade do evento
  - `timex3-temporalfunction:` `temporalFunction` da expressão temporal anotado no *corpus*
  - `timex3-type:` tipo da expressão temporal

## Conhecimento Sobre o Mundo
  - `event-temporal-direction:` registra a direção temporal do evento. Representa a relação temporal esperada entre o evento e seu complemento.
  - `event-closest-to-event-temporal-direction:` valor da direção temporal para o evento mais próximo do evento da relação temporal sob classificação

## Contextual
  - `event-between-order:` se há outro evento entre o evento e a expressão temporal da relação sob classificação
  - `event-conjunction-closest-follow:` conjunção mais próxima após o evento da relação processada
  - `event-conjunction-closest-precede:` conjunção mais próxima antes do evento da relação processada
  - `event-first-order:` o evento precede textualmente a expressão temporal?
  - `event-has-modal-verb-precede:` verifica se o evento possui verbos modais antes dele 
  - `event-modal-verb:` texto do verbo modal antes do evento
  - `event-preposition-precede:` tem como valor a preposição que precede o evento no texto
  - `event-timex3-distance:` distância entre o evento e expressão temporal.
  - `event-timex3-no-between-order:` verifica se não há evento ou expressão temporal entre o evento e a expressão temporal da relação sob classificação 
  - `timex3-between-order:` se há outra expressão temporal entre o evento e a expressão temporal da relação sob classificação
  - `timex3-preposition-precede:` tem como valor a preposição que precede a expressão temporal no texto

## Informações Lexicais
  - `timex3-relevant-lemmas:` o valor do lema da expressão temporal se ele estiver em uma lista de lemas que têm algum conteúdo temporal e são frequentemente vistos nas expressões temporais.

## Informações Morfológicas
  - `event-closest-to-event-equal-pos:` verifica se a classe gramatical do evento da relação temporal sob classificação é igual à classe gramatical do evento que é mencionado no texto mais próximo a ele
  - `event-closest-to-event-equal-tense:` verifica se o tempo verbal do evento da relação temporal sob classificação é igual ao tempo verbal do evento mais próximo a ele no texto
  - `event-closest-to-event-pos:` classe gramatical do evento que está mais próximo do evento que está na relação temporal sob classificação
  - `event-closest-to-event-tense:` tempo verbal do evento que está mais próximo do evento que está na relação temporal sob classificação
  - `event-closest-to-timex3-equal-pos:` verifica se a classe gramatical do evento da relação sob classificação é igual à classe gramatical do evento mais próximo da expressão temporal da relação sob classificação
  - `event-closest-to-timex3-pos:` classe gramatical do evento mais próximo da expressão temporal da relação temporal sob classificação
  - `event-gov-verb-aspect:` aspecto verbal do verbo que rege o evento. Se evento for verbo, o valor dessa *feature* é o aspecto verbal do próprio evento
  - `event-gov-verb-tense:` tempo verbal do verbo que rege o evento. Se evento for verbo, o valor dessa *feature* é o tempo verbal do próprio evento
  - `event-head-pos:` classe gramatical do nó pai do evento na árvore de dependência
  - `event-intervening-following-tense:` tempo verbal do evento que é mencionado entre o evento e a expressão temporal, nesta ordem textual, da relação temporal sob classificação e está mais próximo da expressão temporal.  
     Por exemplo, `EVENT ------------ event2.tense --- TIMEX`.
  - `event-pos:` classe gramatical do evento
  - `event-pos-token-1-follow:` classe gramatical do 1º *token* após o evento
  - `event-pos-token-2-follow:` classe gramatical do 2º *token* após o evento
  - `event-pos-token-3-follow:` classe gramatical do 3º *token* após o evento
  - `event-pos-token-1-precede:` classe gramatical do 1º *token* anterior ao evento
  - `event-pos-token-2-precede:` classe gramatical do 2º *token* anterior ao evento
  - `event-pos-token-3-precede:` classe gramatical do 3º *token* anterior ao evento
  - `timex3-pos-token-1-follow:` classe gramatical do 1º *token* após a expressão temporal
  - `timex3-pos-token-2-follow:` classe gramatical do 2º *token* após a expressão temporal
  - `timex3-pos-token-3-follow:` classe gramatical do 3º *token* após a expressão temporal
  - `timex3-pos-token-1-precede:` classe gramatical do 1º *token* anterior à expressão temporal
  - `timex3-pos-token-2-precede:` classe gramatical do 2º *token* anterior à expressão temporal
  - `timex3-pos-token-3-precede:` classe gramatical do 3º *token* anterior à expressão temporal
  - `timex3-gov-verb-tense:` tempo verbal do verbo que rege a expressão temporal
  - `timex3-head-pos:` classe gramatical do nó pai de expressão temporal
  - `timex3-pos:` classe gramatical da expressão temporal

## Informações Sintáticas
  - `event-dep:` relação de dependência que o evento tem com seu regente?
  - `event-head-is-root:` o evento modifica diretamente a raiz?  
    (ex: o evento é um filho direto da raiz da árvore de dependência?)
  - `event-is-ancestor-timex3:` o evento é a entidade regente na relação?
  - `event-preposition-gov:` preposição que rege sintaticamente o evento
  - `event-root:` o evento é a raiz da árvore de dependência sintática? 
  - `event-timex3-dep:` relação de dependência entre o evento e a expressão temporal ou entre a expressão temporal e o evento 
  - `reichenbach-direct-modification:` a expressão temporal modifica diretamente o evento?  
    (a expressão está no mesmo caminho de dependência do evento?)
  - `reichenbach-temporal-mod-function:` Função de modificação temporal, existe uma relação `nmod` no caminho de dependência do evento à expressão temporal? 
  - `timex3-dep:` relação de dependência sintática da expressão temporal com seu regente (nó pai).
  - `timex3-head-is-root:` a expressão temporal modifica diretamente a raiz?  
    (ex: a expressão temporal é um filho direto da raiz da arvore de dependência)
  - `timex3-is-ancestor-event:` a expressão temporal é a entidade regente na relação?
  - `timex3-preposition-gov:` preposição que rege sintaticamente a expressão temporal
    
## Reichenbach
  - `reichenbach-tense:` tempo verbal de Reichenbach. Assume o valor anterior, simples ou posterior. 
    
## Sinais Temporais
  Os sinais temporais consistem em conjunções e advérbios temporais. É importante mencionar que se não existir sinal que preceda o evento ou a expressão temporal sob classificação, o valor será nulo para todas as features relacionadas a este sinal.
    
  - `signal-precede-event-child-event:` o sinal que precede o evento é regido sintaticamente por este evento?
  - `signal-precede-timex3-child-timex3:` o sinal que precede a expressão temporal é regido sintaticamente por esta expressão?
  - `signal-precede-event-dep-if-child-event:` se o sinal que precede o evento for regido por este evento, qual é o tipo de relação de dependência sintática?
  - `signal-precede-timex3-dep-if-child-timex3:` se o sinal que precede a expressão temporal for regido por esta expressão, qual é o tipo de relação de dependência sintática?
  - `signal-precede-event-distance-event:` distância entre o evento e o sinal temporal que o precede
  - `signal-precede-timex3-distance-timex3:` distância entre a expressão temporal e o sinal temporal que a precede
  - `signal-precede-event-pos:` classe gramatical do sinal que precede o evento
  - `signal-precede-timex3-pos:` classe gramatical do sinal que precede a expressão temporal
  - `signal-precede-event-text:` texto do sinal que precede o evento
  - `signal-precede-timex3-text:` texto do sinal que precede a expressão temporal
