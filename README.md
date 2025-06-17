🏦 Credit Card Approval Classification

Este projeto tem como objetivo desenvolver modelos de aprendizado de máquina para prever a aprovação de cartão de crédito, utilizando algoritmos de Naive Bayes e Perceptron Multicamadas (MLP).

O trabalho faz parte da disciplina Sistemas Inteligentes - INE5633 da Universidade Federal de Santa Catarina (UFSC).

----------------------------------------------------
🚀 Tecnologias Utilizadas

- Python 3.11
- Pandas
- Scikit-Learn
- TensorFlow (Keras)
- Python-docx (para geração de relatórios)

----------------------------------------------------
📂 Estrutura do Projeto

T2-INE5633/
├── data/              -> Dados utilizados (cc_approvals.data)
├── outputs/           -> Resultados gerados (relatórios, matrizes)
├── src/               -> Código fonte
│   └── main.py        -> Script principal
├── README.txt         -> Este arquivo
├── requirements.txt   -> Dependências do projeto
├── .gitignore         -> Arquivos e pastas ignorados no versionamento
└── .python-version    -> Versão do Python usada (gerenciado pelo pyenv)

----------------------------------------------------
📑 Descrição do Dataset

O dataset contém registros de clientes com informações como:

- Gênero
- Estado civil
- Histórico bancário
- Escolaridade
- Pontuação de crédito
- Histórico de emprego
- Renda

A variável alvo é ApprovalStatus, onde:
- "+" -> Aprovado
- "-" -> Não aprovado

Fonte dos dados: Kaggle - Credit Card Approval
Link: https://www.kaggle.com/datasets/youssefaboelwafa/credit-card-approval

----------------------------------------------------
⚙️ Como Executar Localmente

1. Clone o repositório:

git clone https://github.com/seu-usuario/T2-INE5633.git
cd T2-INE5633

2. Configure o ambiente Python:

Usando pyenv (recomendado):

pyenv local 3.11.9
python -m venv venv
source venv/bin/activate

3. Instale as dependências:

pip install -r requirements.txt

4. Execute o script:

python src/main.py

Os resultados serão exibidos no console e também podem ser salvos na pasta outputs/ (se configurado no código).

----------------------------------------------------
📊 Resultados Gerados

- Matrizes de Confusão
- Medidas de desempenho (F1 Score)
- Relatório em .docx comparando os modelos

----------------------------------------------------
🧠 Modelos Desenvolvidos

- Naive Bayes: Algoritmo probabilístico simples, assume independência entre os atributos.
- MLP (Perceptron Multicamadas): Rede neural com múltiplas camadas, capaz de modelar relações não-lineares.

----------------------------------------------------
🚫 Limitações

- Dataset pequeno e simplificado.
- Modelos não otimizados com tuning avançado de hiperparâmetros.

----------------------------------------------------
🏫 Disciplina

- Universidade: Universidade Federal de Santa Catarina (UFSC)
- Curso: Sistemas de Informação
- Disciplina: INE5633 - Sistemas Inteligentes
- Professor: Alexandre Gonçalves Silva

----------------------------------------------------
✍️ Autoria

Projeto desenvolvido por:
- Vanessa Cunha (17100926)

----------------------------------------------------
📜 Licença

Este projeto é acadêmico e não possui fins comerciais.
