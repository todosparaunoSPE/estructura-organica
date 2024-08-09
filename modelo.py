# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 09:00:18 2024

@author: jperezr
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# --- Funciones ---
def generar_datos_empleados(num_empleados=200):
    """
    Genera un archivo .xlsx con datos ficticios de empleados.
    """
    data = {
        'ID': np.arange(1, num_empleados + 1),
        'Nombre': [f'Empleado {i}' for i in range(1, num_empleados + 1)],
        'Evaluacion_Desempeno': np.random.randint(1, 6, num_empleados),
        'Retroalimentacion': np.random.choice(['Positiva', 'Neutral', 'Negativa'], num_empleados),
        'Habilidades': np.random.choice(['Python', 'Excel', 'Comunicación', 'Liderazgo', 'Gestión de proyectos'], num_empleados),
        'Trayectoria_Laboral': np.random.randint(1, 20, num_empleados),
        'Preferencias_Desarrollo': np.random.choice(['Gerencia', 'Especialización', 'Cambio de rol', 'Desarrollo técnico'], num_empleados),
        'Departamento': np.random.choice(['Ventas', 'Marketing', 'TI', 'Recursos Humanos', 'Finanzas'], num_empleados)
    }
    df = pd.DataFrame(data)
    df.to_excel('empleados.xlsx', index=False)
    return df

# --- Configuración Inicial ---
st.title("Desarrollo Profesional Personalizado con IA en PENSIONISSSTE")

# --- Barra Lateral - Ayuda ---
with st.sidebar:
    st.header("Ayuda")
    st.write("""
    **Desarrollo Profesional Personalizado con IA en PENSIONISSSTE**

    Esta aplicación tiene como objetivo proporcionar una plataforma integral para la gestión del desarrollo profesional dentro de PENSIONISSSTE. Utilizando técnicas avanzadas de inteligencia artificial, la aplicación permite:

    1. **Generación y Visualización de Datos**: Creación de un archivo de datos ficticios de empleados que incluye información sobre evaluaciones de desempeño, retroalimentación, habilidades y trayectoria laboral.

    2. **KPIs y Métricas Clave**: Evaluación del desempeño general de los empleados, distribución de retroalimentación y análisis de habilidades por departamento.

    3. **Comparativas Departamentales**: Comparación detallada del desempeño y la trayectoria laboral entre diferentes departamentos, así como la distribución de habilidades.

    4. **Filtros Interactivos**: Permite a los usuarios aplicar filtros para analizar datos específicos de empleados basados en departamento y habilidades.

    5. **Panel de Aprobación de Recomendaciones**: Facilita la aprobación, modificación o rechazo de recomendaciones de desarrollo profesional para empleados filtrados.

    6. **Asignación de Recursos**: Permite la asignación de recursos adicionales a empleados con evaluaciones de desempeño altas.

    7. **Retroalimentación y Comentarios**: Captura retroalimentación y comentarios directos sobre empleados y guarda esta información en un archivo para futuras referencias.

    8. **Simulación de Impacto de Decisiones**: Simula el impacto de decisiones de cambio de rol en la evaluación de desempeño de los empleados.

    9. **Modelo de Predicción de Retroalimentación**: Utiliza un modelo de aprendizaje automático para predecir la retroalimentación futura basada en la evaluación de desempeño y la trayectoria laboral de los empleados.

    **Importancia del Desarrollo Profesional Personalizado:**

    El desarrollo profesional personalizado es crucial para maximizar el potencial de cada empleado y asegurar que los recursos de la organización se utilicen de manera eficiente. Con la ayuda de la inteligencia artificial, se pueden tomar decisiones basadas en datos para mejorar el desempeño y la satisfacción de los empleados, lo que a su vez contribuye al éxito general de PENSIONISSSTE.
    """)

# Verificar si el archivo existe
if os.path.exists('empleados.xlsx'):
    df = pd.read_excel('empleados.xlsx')
    if 'Departamento' not in df.columns:
        st.error("El archivo 'empleados.xlsx' no contiene la columna 'Departamento'.")
        st.stop()
else:
    df = generar_datos_empleados()
    st.write("Archivo 'empleados.xlsx' generado.")

# --- KPIs y Métricas Clave ---
st.header("KPIs y Métricas Clave")

# KPI 1: Evaluación Promedio de Desempeño
eval_promedio = df['Evaluacion_Desempeno'].mean()
st.metric(label="Evaluación Promedio de Desempeño", value=f"{eval_promedio:.2f}")

# KPI 2: Número de Empleados por Departamento
num_empleados_dept = df['Departamento'].value_counts()
st.bar_chart(num_empleados_dept, use_container_width=True)

# KPI 3: Distribución de Retroalimentación
retroalimentacion_dist = df['Retroalimentacion'].value_counts()
st.write("Distribución de Retroalimentación")
st.bar_chart(retroalimentacion_dist, use_container_width=True)

# --- Comparativas Departamentales ---
st.header("Comparativas Departamentales")

# Comparativa de Evaluaciones de Desempeño por Departamento
fig1 = px.box(df, x='Departamento', y='Evaluacion_Desempeno', title='Distribución de Evaluaciones de Desempeño por Departamento')
st.plotly_chart(fig1)

# Comparativa de Trayectoria Laboral por Departamento
fig2 = px.box(df, x='Departamento', y='Trayectoria_Laboral', title='Trayectoria Laboral por Departamento')
st.plotly_chart(fig2)

# Comparativa de Habilidades por Departamento
fig3 = px.histogram(df, x='Habilidades', color='Departamento', title='Distribución de Habilidades por Departamento')
st.plotly_chart(fig3)

# --- Filtro Interactivo ---
st.sidebar.header("Filtros Interactivos")
dept_filter = st.sidebar.multiselect("Selecciona Departamento", options=df['Departamento'].unique(), default=df['Departamento'].unique())
habilidades_filter = st.sidebar.multiselect("Selecciona Habilidades", options=df['Habilidades'].unique(), default=df['Habilidades'].unique())

df_filtered = df[(df['Departamento'].isin(dept_filter)) & (df['Habilidades'].isin(habilidades_filter))]

st.subheader("Datos Filtrados")
st.write(df_filtered)

# --- Panel de Aprobación de Recomendaciones ---
st.header("Panel de Aprobación de Recomendaciones")
for index, row in df_filtered.iterrows():
    st.subheader(f"{row['Nombre']} ({row['Departamento']})")
    st.write(f"Evaluación de Desempeño: {row['Evaluacion_Desempeno']}")
    st.write(f"Trayectoria Laboral: {row['Trayectoria_Laboral']} años")
    st.write(f"Preferencias de Desarrollo: {row['Preferencias_Desarrollo']}")

    aprobacion = st.selectbox(
        f"¿Aprobar recomendación para {row['Nombre']}?", 
        ["Aprobar", "Modificar", "Rechazar"], 
        key=f"aprobacion_{row['ID']}_{index}"
    )

    if aprobacion == "Modificar":
        nueva_recomendacion = st.text_input(f"Modificar recomendación para {row['Nombre']}", key=f"modificar_{row['ID']}_{index}")
    elif aprobacion == "Rechazar":
        st.warning(f"Recomendación rechazada para {row['Nombre']}.")
    else:
        st.success(f"Recomendación aprobada para {row['Nombre']}.")

# --- Asignación de Recursos Basada en Evaluaciones ---
st.header("Asignación de Recursos")
for index, row in df_filtered.iterrows():
    st.subheader(f"{row['Nombre']} ({row['Departamento']})")
    st.write(f"Evaluación de Desempeño: {row['Evaluacion_Desempeno']}")
    st.write(f"Trayectoria Laboral: {row['Trayectoria_Laboral']} años")
    st.write(f"Preferencias de Desarrollo: {row['Preferencias_Desarrollo']}")

    if row['Evaluacion_Desempeno'] >= 4:
        asignar_recursos = st.checkbox(
            f"Asignar recursos para {row['Nombre']} ({row['Departamento']})", 
            key=f"asignar_recursos_{row['ID']}_{index}"
        )
        
        if asignar_recursos:
            st.success(f"Recursos asignados a {row['Nombre']}.")
    else:
        st.warning(f"Desempeño insuficiente para asignación de recursos adicionales.")

# --- Retroalimentación y Comentarios Directos ---
st.header("Retroalimentación Directa")
feedback_data = []

for index, row in df_filtered.iterrows():
    st.subheader(f"{row['Nombre']} ({row['Departamento']})")
    st.write(f"Evaluación de Desempeño: {row['Evaluacion_Desempeno']}")
    st.write(f"Trayectoria Laboral: {row['Trayectoria_Laboral']} años")
    st.write(f"Preferencias de Desarrollo: {row['Preferencias_Desarrollo']}")

    comentario = st.text_area(f"Comentarios de {row['Nombre']}", key=f"comentario_{row['ID']}_{index}")
    feedback_data.append({
        'ID': row['ID'],
        'Nombre': row['Nombre'],
        'Comentario': comentario
    })

feedback_df = pd.DataFrame(feedback_data)
if not feedback_df.empty:
    st.write("Comentarios Recibidos:")
    st.write(feedback_df)

# --- Simulación de Impacto de Decisiones ---
st.header("Simulación de Impacto de Decisiones")
st.write("Simula cómo las decisiones de cambio de rol pueden impactar la evaluación de desempeño de los empleados.")

# Simulación usando un modelo de IA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Preparar datos para el modelo
X = df[['Evaluacion_Desempeno', 'Trayectoria_Laboral']].values
y = df['Retroalimentacion'].apply(lambda x: 1 if x == 'Positiva' else (2 if x == 'Neutral' else 3)).values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluación del modelo
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
st.write(f"Precisión del Modelo de IA: {accuracy:.2f}")

# Entradas del usuario para predicción
st.sidebar.header("Entradas del Usuario")
eval_input = st.slider("Evaluación de Desempeño", min_value=1, max_value=5, value=3)
trayectoria_input = st.slider("Trayectoria Laboral", min_value=1, max_value=20, value=5)

# Predicción
prediccion = model.predict([[eval_input, trayectoria_input]])[0]
retroalimentacion = {1: 'Positiva', 2: 'Neutral', 3: 'Negativa'}
st.write(f"La retroalimentación predicha es: {retroalimentacion[prediccion]}")