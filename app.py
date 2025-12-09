# Importamos las librerías necesarias.
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
# Agregamos un título de pestaña con streamlit.
st.set_page_config(page_title="Sistema de Gestión Académica", layout="wide")
# Creamos un título principal y una descripción.
st.title("Sistema de Análisis de Rendimiento y Asistencia")
st.markdown("""
Plataforma de inteligencia de negocios para el análisis de deserción y rendimiento académico.
Utilice los filtros laterales para segmentar por carrera.
""")

@st.cache_data
def cargar_datos():
    try:
        # Cargamos archivos .csv.
        df_egresados = pd.read_csv('Base de Datos - Perfil de egresados (1).csv', sep=';', encoding='latin-1')
        df_motivacion = pd.read_csv('Cuestionario motivacion academica.csv', encoding='utf-8')
        df_facultad = pd.read_csv('Data_UINN_Facultad.csv', sep=';', encoding='utf-8-sig')
        
        # Asignamos la tercera fila como nombres de columnas en df_facultad.
        df_facultad.columns = df_facultad.iloc[2]
        # Eliminar las primeras tres filas, las primeras dos son metadatos y la tercera ya se usó como encabezados. Así lograremos visualizar de manera correcta las tablas.
        df_facultad = df_facultad.drop([0, 1, 2]).reset_index(drop=True)
        
        # Normalización de puntajes
        if df_facultad['Puntaje Ponderado'].dtype == object:
             df_facultad['Puntaje Ponderado'] = df_facultad['Puntaje Ponderado'].astype(str).str.replace(',', '.', regex=False)
        df_facultad['Puntaje Ponderado'] = pd.to_numeric(df_facultad['Puntaje Ponderado'], errors='coerce')
        
        df_facultad.dropna(subset=['Puntaje Ponderado'], inplace=True)
        
        # Merge de los DataFrames df_facultad, df_motivacion y df_egresados, utilizamos .concat ya que no logramosencontar una columna en común para realizar nuestro merge de datos.
        df_completo = pd.concat([df_facultad, df_motivacion, df_egresados], axis=1)
        
        # En esta sección, reemplazamos los valores nulos del DataFrame ahora llamado df_completo con la moda de cada columna, y de esta manera no borramos datos y obtenemos un conjunto de datos más completo para nuestro análisis.
        for column in df_completo.columns:
            df_completo[column] = df_completo[column].fillna(df_completo[column].mode()[0])
        
        # Creamos una tabla más pequeña con la finalidad de facilitar nuestro trabajo al graficar y analizar datos específicos.
        df_small = df_completo.iloc[:, [15, 20, 21, 61, 3, 9, 19, 22, 46]].copy() 
        
        df_small.columns = [
            'Carrera', 'Motivacion', 'Asistencia', 'Duracion_Carrera', 
            'Sexo', 'Puntaje_Ponderado', 'Reprobadas', 'Participacion', 'Abandono'
        ]
        
        # Convertiremos columnas específicas a datos númericos.
        cols_numericas = ['Motivacion', 'Asistencia', 'Participacion', 'Reprobadas', 'Carrera']
        for col in cols_numericas:
            df_small[col] = pd.to_numeric(df_small[col], errors='coerce')
            
        # Creamos una nueva columna llamada Estado, la cual indica si el estudiante aprueba o reprueba según la cantidad de asignaturas reprobadas y así buscando la creación de un futuro modelo. Si en la columna Reprobadas el valor es 0, entonces el estudiante aprueba, de lo contrario reprueba.
        df_small["Estado"] = df_small["Reprobadas"].apply(lambda x: 0 if x == 0 else 1) 
        
        return df_small
    except Exception as e:
        st.error(f"Error crítico en la carga de datos: {e}")
        return None

df_global = cargar_datos()

if df_global is None:
    st.stop()

# Definimos nuestro sidebar
st.sidebar.header("Panel de Control")

# Se crea el dominio para cada carrera mediante su código respectivo
carreras_unicas = sorted(df_global['Carrera'].unique())
carreras_selec = st.sidebar.multiselect(
    "Filtrar por Código de Carrera:",
    options=carreras_unicas,
    default=carreras_unicas
)

# Se aplica el filtro, de manera que si se hace una selección de carreras el df_global se reduzca al df_filtrado
if not carreras_selec:
    df_filtrado = df_global.copy()
else:
    df_filtrado = df_global[df_global['Carrera'].isin(carreras_selec)]

# Implementamos dos métricas (Total de Estudiantes y Tasa de Riesgo) calculadas al momento de filtrar
st.sidebar.markdown("---")
st.sidebar.subheader("Métricas del Filtro")
total_estudiantes = len(df_filtrado)
if total_estudiantes > 0:
    tasa_reprobacion = (df_filtrado['Estado'].sum() / total_estudiantes) * 100
    st.sidebar.metric("Total Estudiantes", total_estudiantes)
    st.sidebar.metric("Tasa de Riesgo", f"{tasa_reprobacion:.1f}%")
else:
    st.sidebar.warning("Sin datos para mostrar.")

# X será nuestra variable predictora, que contiene Motivación y Asistencia.
X = df_global[["Motivacion", "Asistencia"]]
# Y será nuestra variable objetivo, que indica si el estudiante aprueba o reprueba.
y = df_global["Estado"]
# Creamos un modelo de regresión logística para predecir si un estudiante aprueba o reprueba basado en su motivación y asistencia.
modelo = LogisticRegression()
modelo.fit(X, y)

# Enseñamos los coeficientes del modelo
coef_mot, coef_asist = modelo.coef_[0]
intercepto = modelo.intercept_[0]

# Hacemos uso de statsmodel para saber si las variables utilizadas son significativas (p-valor)
X_stats = sm.add_constant(X)
modelo_stat = sm.Logit(y, X_stats)
# Ponemos un try/except por si son muy pocos datos o si la varianza es cero para evitar errores matemáticos
try:
    resultado_stat = modelo_stat.fit(disp=0)
    p_values = resultado_stat.pvalues
except:
    p_values = None

# Se crean tres pestañas para mejorar la interactividad de la página
tab1, tab2, tab3 = st.tabs(["Panorama General", "Modelo Predictivo", "Base de Datos"])

# Desarrollo de la primera pestaña "Panorama General", la cual muestra todos los gráficos en función de las carreras seleccionadas
with tab1:
    st.header("Análisis Exploratorio de Datos")
    
    if df_filtrado.empty:
        st.warning("Seleccione al menos una carrera para visualizar datos.")
    else:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Matriz de Correlaciones")
            cols_corr = ['Motivacion', 'Asistencia', 'Participacion', 'Reprobadas', 'Puntaje_Ponderado']
            corr = df_filtrado[cols_corr].corr()
            fig_corr, ax = plt.subplots(figsize=(6, 4))
            sns.heatmap(corr, annot=True, cmap='coolwarm', linewidths=0.5, ax=ax)
            st.pyplot(fig_corr)

        with col2:
            st.subheader("Impacto: Asistencia vs Reprobación")
            fig_scat, ax = plt.subplots(figsize=(6, 4))
            sns.scatterplot(data=df_filtrado, x="Asistencia", y="Reprobadas", hue="Estado", palette="coolwarm", ax=ax)
            ax.set_xlabel("Nivel de Asistencia (1-5)")
            ax.set_ylabel("Asignaturas Reprobadas")
            st.pyplot(fig_scat)

        st.markdown("---")
        st.subheader("Distribución de Reprobación por Carrera")
        fig_blue, ax = plt.subplots(figsize=(10, 4))
        ax.scatter(df_filtrado['Carrera'], df_filtrado['Reprobadas'], alpha=0.6, color='blue')
        ax.set_title("Dispersión: Código de Carrera vs. Cantidad de Reprobadas")
        ax.set_xlabel("Código de Carrera")
        ax.set_ylabel("Cantidad de Asignaturas Reprobadas")
        ax.grid(True, linestyle='--', alpha=0.5)
        st.pyplot(fig_blue)

# Desarrollo de la segunda pestaña "Análisis Predictivo"
with tab2:
    st.header("Análisis Predictivo")
    
    # Se añade el recuadro para la visualización de los valores-p en dicho apartado
    st.subheader("Validación Estadística (Significancia)")
    st.write("El valor-P indica si la variable es estadísticamente relevante. Si es < 0.05, la relación es significativa.")
    
    if p_values is not None:
        col_p1, col_p2 = st.columns(2)
        
        # Se asigna el valor-p correspondiente a la motivación
        with col_p1:
            p_val_mot = p_values["Motivacion"]
            # Se aplica representación de valores demasiado pequeños
            st.metric("Valor-P Motivación", f"{p_val_mot:.2e}") 
            if p_val_mot < 0.05:
                st.success("Variable Significativa (Relación Fuerte)")
            else:
                st.error("Variable No Significativa")

        # Se asigna el valor-p correspondiente a la asistencia
        with col_p2:
            p_val_asist = p_values["Asistencia"]
            # Al igual que en la variable anterior, se aplica representación de valores demasiado pequeños
            st.metric("Valor-P Asistencia", f"{p_val_asist:.2e}")
            if p_val_asist < 0.05:
                st.success("Variable Significativa (Relación Fuerte)")
            else:
                st.error("Variable No Significativa")
    else:
        st.warning("No se pudo calcular el valor-P con los datos actuales.")

    st.markdown("---")

    # Se presentan los coeficientes de la regresión lineal
    st.subheader("Parámetros del Modelo")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Coef. Motivación", f"{coef_mot:.3f}")
    m2.metric("Coef. Asistencia", f"{coef_asist:.3f}")
    m3.metric("Intercepto", f"{intercepto:.3f}")
    
    y_pred = modelo.predict(X)
    acc = accuracy_score(y, y_pred)
    m4.metric("Exactitud Global", f"{acc*100:.1f}%")

    # Se implementan sliders, permitiendo personalizar el rango de motivación y asistencia del estudiante
    st.markdown("---")
    st.header("Simulador de Riesgo Individual")
    
    input_c1, input_c2, res_c = st.columns([1, 1, 2])
    with input_c1:
        val_mot = st.slider("Nivel de Motivación", 1, 5, 3)
    with input_c2:
        val_asist = st.slider("Nivel de Asistencia", 1, 5, 3)
        
    score = intercepto + (coef_mot * val_mot) + (coef_asist * val_asist)
    
    with res_c:
        st.subheader("Diagnóstico:")
        if score <= 0:
            st.success(f"BAJO RIESGO (Score: {score:.2f}) - Probable Aprobación")
        elif score <= 5:
            st.warning(f"RIESGO MODERADO (Score: {score:.2f}) - Requiere Seguimiento")
        else:
            st.error(f"ALTO RIESGO (Score: {score:.2f}) - Alerta de Reprobación")

# Desarrollo de la segunda pestaña "Base de Datos Filtrada"
with tab3:
    st.header("Base de Datos Filtrada")
    
    @st.cache_data
    def convert_df(df):
        return df.to_csv(index=False).encode('utf-8')

    if not df_filtrado.empty:
        # Se añade un botón de descarga en pdf (en caso de no haber suficiente memoria en la página para cargar todos los datos filtrados por carrera)
        csv = convert_df(df_filtrado)
        
        st.download_button(
            label=f"Descargar Datos Completos ({len(df_filtrado)} filas)",
            data=csv,
            file_name='reporte_academico.csv',
            mime='text/csv',
        )
        
        # Ajustamos un límite de filas de datos con el fin de no sobrecargar el rendimiento de la página
        LIMITE_FILAS = 1000
        
        if len(df_filtrado) > LIMITE_FILAS:
            st.info(f"Mostrando primeros {LIMITE_FILAS} registros para optimizar rendimiento. Descargue el CSV para ver todo.")
            df_visual = df_filtrado.head(LIMITE_FILAS)
        else:
            df_visual = df_filtrado

        # Se resaltan todas las filas que tengan el estado de riesgo académico
        def highlight_risk(s):
            return ['background-color: #ffcccc' if v == 1 else '' for v in s]

        st.dataframe(
            df_visual.style.apply(highlight_risk, subset=['Estado']),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("No hay datos para mostrar con los filtros actuales.")