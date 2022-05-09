## Text Tokenization

### Sentence Tokenization

Tokenização de sentença é o processo de dividir um corpus de texto em sentenças. Isso também é conhecido como sentença segmentação, porque tentamos segmentar o texto em frases significativas.

```py
import nltk
from pprint import pprint

nltk.download('punkt')

text = 'We will discuss briefly about the basic syntax, structure and\
        design philosophies. There is a defined hierarchical syntax for Python code\
        which you should remember when writing code! Python is a really powerful\
        programming language!'

punkt_st = nltk.tokenize.PunktSentenceTokenizer()
sample_sentences = punkt_st.tokenize(sample_text)

pprint(sample_sentences)
```

### Word Tokenization

Tokenização de palavras é o processo de dividir ou segmentar frases em suas palavras constituintes. A tokenização de palavras é muito importante em muitos processos, especialmente na limpeza e normalização de texto, onde operações como lematização e lematização funcionam em cada palavra individual com base em seus respectivos radicais e lemas.

```py
sentence = "The brown fox wasn't that quick and he couldn't win the race"

default_wt = nltk.word_tokenize
words = default_wt(sentence)

print(words)
```

* Padrão para identificar os **próprios tokens**

```py
sentence = "The brown fox wasn't that quick and he couldn't win the race"

TOKEN_PATTERN = r'\w+'
regex_wt = nltk.RegexpTokenizer(pattern=TOKEN_PATTERN, gaps=False)
words = regex_wt.tokenize(sentence)

print(words)
```

* Padrão para identificar **lacunas em tokens**

```py
sentence = "The brown fox wasn't that quick and he couldn't win the race"

GAP_PATTERN = r'\s+'
regex_wt = nltk.RegexpTokenizer(pattern=GAP_PATTERN, gaps=True)
words = regex_wt.tokenize(sentence)

print(words)
```

* O WhitespaceTokenizer tokeniza frases em palavras com base em espaços em branco como tabulações, novas linhas e espaços .

```py
sentence = "The brown fox wasn't that quick and he couldn't win the race"

whitespace_wt = nltk.WhitespaceTokenizer()
words = whitespace_wt.tokenize(sentence)

print(words)
```

## Text Normalization

### Cleaning Text

Muitas vezes, os dados textuais que queremos usar ou analisar contêm muitas informações estranhas e tokens e caracteres desnecessários que devem ser removidos antes de realizar qualquer operações adicionais como tokenização ou outras técnicas de normalização. Isso inclui extrair texto significativo de fontes de dados como dados HTML, que consiste em tags HTML desnecessárias, ou mesmo dados de feeds XML e JSON.

### Tokenization Text

Normalmente, tokenizamos o texto antes ou depois de remover caracteres e símbolos desnecessários a partir dos dados.

### Removing Special Characters

Uma tarefa importante na normalização de texto envolve a remoção de caracteres desnecessários e especiais. Estes podem ser símbolos especiais ou até mesmo a pontuação que ocorre nas frases. Essa etapa geralmente é executada antes ou depois da tokenização. A principal razão para fazer isso é porque muitas vezes a pontuação ou os caracteres especiais não têm muito significado quando queremos analisar o texto e utilizá-lo para extrair recursos ou informações baseadas em NLP e ML.

```py
corpus = [
  "The brown fox wasn't that quick and he couldn't win the race",\
  "Hey that's a great deal! I just bought a phone for $199",\
  "@@You'll (learn) a **lot** in the book. Python is an amazing\
  language !@@"
]

def remove_characters_before_tokenization(sentence, keep_apostrophes=False):
  sentence = sentence.strip()

  if keep_apostrophes:
    PATTERN = r'[?|$|&|*|%|@|(|)|~]' # add other characters here to remove them
    filtered_sentence = re.sub(PATTERN, r'', sentence)
  else:
    PATTERN = r'[^a-zA-Z0-9 ]' # only extract alpha-numeric characters
    filtered_sentence = re.sub(PATTERN, r'', sentence)

  return filtered_sentence

filtered_list = [remove_characters_before_tokenization(sentence) for sentence in corpus]

print(filtered_list)
```

### Stopwords

Stopwords, às vezes escrito stop words, são palavras que têm pouco ou nenhum significado. Eles geralmente são removidos do texto durante o processamento para reter palavras com significado e contexto máximos. Stopwords geralmente são palavras que acabam ocorrendo mais se você agregar qualquer corpus de texto com base em tokens singulares e verificar suas frequências.

```py
corpus = [
  "The brown fox wasn't that quick and he couldn't win the race",\
  "Hey that's a great deal! I just bought a phone for $199",\
  "@@You'll (learn) a **lot** in the book. Python is an amazing\
  language !@@"
]

def remove_stopwords(tokens):
  stopword_list = nltk.corpus.stopwords.words('english')
  filtered_tokens = [token for token in tokens if token not in stopword_list]

  return filtered_tokens

nltk.download('stopwords')

expanded_corpus_tokens = [tokenize_text(text) for text in cleaned_corpus]
filtered_list = [[remove_stopwords(tokens) for tokens in sentence_tokens] for sentence_tokens in expanded_corpus_tokens]

print(filtered_list)

# stopwords removed
nltk.corpus.stopwords.words('english')
```

### Correcting Words

Um dos principais desafios enfrentados na normalização de texto é a presença de palavras incorretas no texto. A definição de incorreto abrange palavras que apresentam erros ortográficos, bem como palavras com várias letras repetidas que não contribuem muito para sua globalidade.