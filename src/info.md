# Conceitos fundamentais

Nós/Vértices : O nó é o círculo que representa cada um dos indivíduos da rede. Nas nossas visualizações, seu tamanho e coloração são alteráveis de acordo com a métrica e o agrupamento (por guilda, cliente, etc.) escolhido.

Laços/Arestas: É a relação estabelecida entre dois indivíduos, que conecta seus respectivos nós. Na nossa rede, os laços podem ser unidirecionais ou recíprocos, isto é, quando um indivíduo reporta estabelecer relação com outro, a recíproca pode ou não ser verdadeira.

Dimensão: Foram chamadas de dimensões as quatro analisadas pelo questionário da ONA: Metodológica, Negócios, Diversos, Controle.

## Métricas de rede

### Métricas de centralidade individuais

Degree centrality: É o número de relações estabelecidas pelo indivíduo, independentemente da direcionalidade, isto é, se é ele quem reporta ter buscado algum membro da rede ou o contrário, para tratar de uma determinada dimensão.

Degree médio: É o número médio de relações por indivídios em um rede, independente de ser relações de entrada e saída.

In degree: É o número de relações que chegam ao indivíduo; isto é, o número de membros da rede que disseram tê-lo buscado para discutir sobre a dimensão analisada.

Out degree: É o número de relações que saem do indivíduo; isto é, o número de membros da rede que ele reportou ter buscado para tratar de assuntos da dimensão analisada.

Betweenness: É a capacidade do indivíduo de controlar o fluxo informacional numa determinada rede. Quanto maior, maior o seu potencial intermediador na rede para os assuntos relacionados a dimensão escolhida.

Eigenvector Centrality: Indivíduos com altos valores de Eigenvector Centrality são aqueles que estabelecem relações com os mais conectados da rede escolhida, ainda que eles próprios possam não ter tantas conexões.

Closeness: É a proximidade do indivíduo em relação aos outros. Quanto mais elevado, mais central, e mais rapidamente as informações sobre a dimensão escolhida se espalham a partir do momento que tem contato com ele.

Constraint: É uma medida de quanto as conexões (ou 'alters') de um indivíduo estão interconectadas entre si. Quanto maior o constraint, mais densamente interligados estão os alters do indivíduo, e menos o indivíduo é capaz de ocupar uma posição de intermediação ou ponte entre grupos distintos em sua rede. Assim, um alto constraint pode limitar a capacidade de um indivíduo de obter vantagens informacionais e estratégicas de sua posição na rede.

### Métricas de centralidade de grupo

Densidade: Diz respeito ao quão integrada uma determinada rede é. É esperado que redes menores sejam mais densas, pois o custo de estabelecer relações com mais indivíduos aumenta conforme o tamanho da rede também aumenta.

Diâmetro: É a distância mais curta entre os dois nós mais distantes da rede no componente do maior componente de rede. Valores mais altos para esta métrica indicam redes maiores e/ou menos integradas. Para o caso das redes com partes não conectadas entre si, é calculado o diâmentro do maior parte ou componente.

Métricas individuais médias: Todas as métricas tratadas no tópico anterior são válidas como indicadores de centralidade de grupo, e estão presentes na aba “Análises de grupo”.
