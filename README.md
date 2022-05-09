## Text Tokenization

### Sentence Tokenization

Tokenização de sentença é o processo de dividir um corpus de texto em sentenças. Isso também é conhecido como sentença segmentação, porque tentamos segmentar o texto em frases significativas.

### Word Tokenization

Tokenização de palavras é o processo de dividir ou segmentar frases em suas palavras constituintes. A tokenização de palavras é muito importante em muitos processos, especialmente na limpeza e normalização de texto, onde operações como lematização e lematização funcionam em cada palavra individual com base em seus respectivos radicais e lemas.

## Text Normalization

### Cleaning Text

Muitas vezes, os dados textuais que queremos usar ou analisar contêm muitas informações estranhas e tokens e caracteres desnecessários que devem ser removidos antes de realizar qualquer operações adicionais como tokenização ou outras técnicas de normalização. Isso inclui extrair texto significativo de fontes de dados como dados HTML, que consiste em tags HTML desnecessárias, ou mesmo dados de feeds XML e JSON.

### Tokenization Text

Normalmente, tokenizamos o texto antes ou depois de remover caracteres e símbolos desnecessários a partir dos dados.

### Removing Special Characters

Uma tarefa importante na normalização de texto envolve a remoção de caracteres desnecessários e especiais. Estes podem ser símbolos especiais ou até mesmo a pontuação que ocorre nas frases. Essa etapa geralmente é executada antes ou depois da tokenização. A principal razão para fazer isso é porque muitas vezes a pontuação ou os caracteres especiais não têm muito significado quando queremos analisar o texto e utilizá-lo para extrair recursos ou informações baseadas em NLP e ML.

### Stopwords

Stopwords, às vezes escrito stop words, são palavras que têm pouco ou nenhum significado. Eles geralmente são removidos do texto durante o processamento para reter palavras com significado e contexto máximos. Stopwords geralmente são palavras que acabam ocorrendo mais se você agregar qualquer corpus de texto com base em tokens singulares e verificar suas frequências.

### Correcting Words

Um dos principais desafios enfrentados na normalização de texto é a presença de palavras incorretas no texto. A definição de incorreto abrange palavras que apresentam erros ortográficos, bem como palavras com várias letras repetidas que não contribuem muito para sua globalidade.