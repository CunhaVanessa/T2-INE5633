import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.preprocessing import OrdinalEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import RMSprop

# Carregar dados
columns = [
    'Gender', 'Age', 'Debt', 'Married', 'BankCustomer', 'EducationLevel',
    'Ethnicity', 'YearsEmployed', 'PriorDefault', 'Employed', 'CreditScore',
    'DriversLicense', 'Citizen', 'ZipCode', 'Income', 'ApprovalStatus'
]
df = pd.read_csv('data/cc_approvals.data', header=None, names=columns)

# Remover coluna irrelevante
df = df.drop(columns=['ZipCode'])

# Codificar variável alvo
df['ApprovalStatus'] = df['ApprovalStatus'].map({'+': 1, '-': 0})

# Identificar colunas categóricas
categorical_columns = ['Gender', 'Married', 'BankCustomer', 'EducationLevel',
                        'Ethnicity', 'PriorDefault', 'Employed', 'DriversLicense', 'Citizen']

# Codificação automática
encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
df[categorical_columns] = encoder.fit_transform(df[categorical_columns])

# Tratar colunas numéricas
for col in ['Age', 'Debt', 'YearsEmployed', 'CreditScore', 'Income']:
    df[col] = pd.to_numeric(df[col], errors='coerce')
    df[col] = df[col].fillna(df[col].mean())

# Separar features e target
X = df.drop(columns=['ApprovalStatus'])
y = df['ApprovalStatus']

# Split treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42
)

# Naive Bayes
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
y_pred_nb = nb_model.predict(X_test)
cm_nb = confusion_matrix(y_test, y_pred_nb)
f1_nb = f1_score(y_test, y_pred_nb)

# MLP
mlp = Sequential([
    Dense(32, activation='tanh', input_shape=(X_train.shape[1],)),
    Dense(16, activation='elu'),
    Dense(8, activation='softplus'),
    Dense(1, activation='sigmoid')
])

mlp.compile(optimizer=RMSprop(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy'])

mlp.fit(X_train, y_train, epochs=30, batch_size=10, validation_split=0.1, verbose=1)

y_pred_mlp = (mlp.predict(X_test) > 0.5).astype(int).reshape(-1)
cm_mlp = confusion_matrix(y_test, y_pred_mlp)
f1_mlp = f1_score(y_test, y_pred_mlp)

# Resultados
print("\nMatriz de Confusão - Naive Bayes:")
print(cm_nb)

print("\nMatriz de Confusão - MLP:")
print(cm_mlp)

print("\nTabela Comparativa de F1 Score:")
print("Modelo         | Medida F1")
print(f"Naive Bayes    | {f1_nb:.4f}")
print(f"MLP            | {f1_mlp:.4f}")
