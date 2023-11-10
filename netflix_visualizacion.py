import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
import sys
import logging


def load_data(filepath):
    """Carga los datos desde un archivo CSV."""
    try:
        df = pd.read_csv(filepath)
        logging.info("Data loaded successfully.")
        return df
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        sys.exit(1)

def preprocess_data(df):
    """Preprocesa los datos para la visualización."""
    print(df.info())
    print(df.head())
    scaler = MinMaxScaler()
    # Paso 2: Tratamiento de valores faltantes
    # Para este ejemplo, vamos a reemplazar valores faltantes en 'country' por una categoría 'Unknown'
    # y en 'director' y 'cast' vamos a eliminar las filas que tengan valores nulos dado que son importantes para el análisis
    df['country'].fillna('Unknown', inplace=True)
    df.dropna(subset=['director', 'cast'], inplace=True)
    # Paso 3: Unificación de tipos de datos
    # Convertir 'date_added' a datetime
    df['date_added'] = pd.to_datetime(df['date_added'], errors='coerce')
    df['year_added'] = df['date_added'].dt.year
    # Paso 4: Estandarización de texto
    # Asumiendo que queremos estandarizar la columna 'title' a minúsculas
    df['title'] = df['title'].str.lower()
    # Paso 5: Manejo de valores atípicos
    # Aquí asumiremos que 'release_year' puede tener valores atípicos y los trataremos
    Q1 = df['release_year'].quantile(0.25)
    Q3 = df['release_year'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df = df[(df['release_year'] >= lower_bound) & (df['release_year'] <= upper_bound)]
    # Paso 6: Normalización/Escalado de Datos
    # Escalaremos 'release_year' usando MinMaxScaler para normalizar los datos entre 0 y 1
    df.loc[:, 'release_year_scaled'] = scaler.fit_transform(df[['release_year']])
    # Limpieza y preparación de datos
    # Paso 7: Codificación de Variables Categóricas
    # Codificaremos 'rating' con One-Hot Encoding
    encoder = OneHotEncoder(sparse_output=False)
    rating_encoded = encoder.fit_transform(df[['rating']])
    rating_encoded_df = pd.DataFrame(rating_encoded, columns=encoder.get_feature_names_out(['rating']))
    df = df.reset_index(drop=True)  # Resetear el índice para evitar problemas de alineación
    df = pd.concat([df, rating_encoded_df], axis=1)
    # Paso 8: Eliminación de Duplicados
    netflix_df = df.drop_duplicates()
    # Paso 9: Ingeniería de Características
    # Como ejemplo, crearemos una nueva característica que es la longitud de la descripción
    df['description_length'] = df['description'].str.len()
    # Paso 10: División de Datos
    # Supongamos que 'type' es nuestra variable objetivo y queremos predecir si un título es una película o un show de TV
    X = df.drop(['type'], axis=1)  # Eliminamos la variable objetivo del conjunto de características
    y = df['type']  # Variable objetivo
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return df

def plot_content_distribution(df):
    """Genera la gráfica de distribución del tipo de contenido."""
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x='type')
    plt.title('Distribución del Tipo de Contenido en Netflix')
    plt.xlabel('Tipo de Contenido')
    plt.ylabel('Cantidad')
    plt.show()

def plot_rating_distribution(df):
    """Genera la gráfica de distribución de ratings."""
    plt.figure(figsize=(10, 8))
    order = df['rating'].value_counts().index
    sns.countplot(data=df, y='rating', order=order)
    plt.title('Distribución de Ratings en Netflix')
    plt.xlabel('Cantidad')
    plt.ylabel('Rating')
    plt.show()

def plot_yearly_content(df):
    """Genera la gráfica de contenido agregado por año."""
    plt.figure(figsize=(12, 8))
    sns.countplot(data=df, x='year_added', color='skyblue')
    plt.title('Cantidad de Contenido Agregado por Año en Netflix')
    plt.xlabel('Año')
    plt.ylabel('Cantidad de Contenido')
    plt.xticks(rotation=45)
    plt.show()

def plot_movie_duration(df):
    """Genera la gráfica de duración de las películas."""
    movies_df = df[df['type'] == 'Movie'].copy()
    movies_df['duration'] = pd.to_numeric(movies_df['duration'].str.replace(' min', ''), errors='coerce')
    movies_df.dropna(subset=['duration'], inplace=True)
    plt.figure(figsize=(12, 8))
    sns.histplot(movies_df['duration'].astype(int), bins=30, kde=True)
    plt.title('Distribución de la Duración de las Películas en Netflix')
    plt.xlabel('Duración (minutos)')
    plt.ylabel('Cantidad de Películas')
    plt.show()

def main(filepath):
    """Función principal del script."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    df = load_data(filepath)
    df = preprocess_data(df)
    plot_content_distribution(df)
    plot_rating_distribution(df)
    plot_yearly_content(df)
    plot_movie_duration(df)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 script.py \\wsl.localhost\\Ubuntu\\home\\jonathan\\netflix_cleaned.csv")
        sys.exit(1)
    main(sys.argv[1])
