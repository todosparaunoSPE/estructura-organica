# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 08:52:44 2024

@author: jperezr
"""

import streamlit as st

# Título de la página
st.title("Desarrollo Profesional Personalizado con IA en PENSIONISSSTE")

# Objetivo del Proyecto
st.header("Objetivo del Proyecto")
st.write("""
Desarrollar un sistema de IA que permita la creación de planes de desarrollo profesional personalizados 
para los empleados de PENSIONISSSTE. El sistema identificará fortalezas y áreas de mejora en cada empleado 
y sugerirá capacitaciones o cambios de rol adecuados para maximizar su potencial y alinearlos con las necesidades organizacionales.
""")

# Fases del Proyecto
st.header("Fases del Proyecto")

st.subheader("1. Análisis de Requerimientos")
st.write("""
- **Identificación de Datos**: Recopilar datos relevantes sobre los empleados, como evaluaciones de desempeño, 
  retroalimentación de superiores y colegas, habilidades actuales, trayectoria laboral, y preferencias de desarrollo profesional.
- **Entrevistas con Stakeholders**: Reunirse con directores de departamentos y recursos humanos para entender las 
  necesidades de desarrollo de talento y los roles clave dentro de la organización.
- **Definición de KPIs**: Establecer indicadores clave de rendimiento (KPIs) para medir el éxito del proyecto, 
  como la tasa de satisfacción de los empleados, la retención de talento, y la eficiencia en el uso de recursos de capacitación.
""")

st.subheader("2. Diseño del Sistema de IA")
st.write("""
- **Modelo de Análisis de Competencias**: Diseñar un modelo que evalúe las competencias y habilidades de cada empleado basado en los datos recopilados.
- **Clasificación de Roles y Competencias**: Crear una base de datos que categorice los roles dentro de la organización y las competencias necesarias para cada uno.
- **Algoritmo de Recomendación**: Desarrollar un algoritmo que sugiera capacitaciones, cambios de rol o mentores basados en la comparación de las competencias actuales del empleado con las requeridas para roles futuros.
""")

st.subheader("3. Desarrollo del Sistema")
st.write("""
- **Integración de Datos**: Implementar un sistema que integre datos de diversas fuentes (evaluaciones, HRIS, encuestas) en un formato centralizado y accesible para el modelo de IA.
- **Entrenamiento del Modelo de IA**: Entrenar el modelo con datos históricos para que aprenda a identificar patrones en las evaluaciones de desempeño y recomendar acciones específicas.
- **Interfaz de Usuario**: Crear una interfaz amigable para que los gerentes y empleados puedan interactuar con el sistema, visualizar sus planes de desarrollo, y recibir recomendaciones de manera intuitiva.
""")

st.subheader("4. Implementación Piloto")
st.write("""
- **Prueba en un Departamento**: Implementar el sistema de manera piloto en un departamento específico para evaluar su efectividad y realizar ajustes.
- **Feedback y Ajustes**: Recopilar feedback de los usuarios del piloto y ajustar el modelo y la interfaz según las necesidades reales.
""")

st.subheader("5. Despliegue Completo")
st.write("""
- **Escalamiento del Sistema**: Desplegar el sistema en toda la organización tras el éxito de la prueba piloto.
- **Capacitación de Usuarios**: Ofrecer capacitaciones a gerentes y empleados sobre cómo utilizar el sistema y entender las recomendaciones generadas.
- **Monitoreo y Mejora Continua**: Establecer un proceso de monitoreo continuo para ajustar el modelo de IA y mejorar la calidad de las recomendaciones basadas en los resultados observados.
""")

st.subheader("6. Evaluación y Reporte")
st.write("""
- **Medición de Resultados**: Evaluar el impacto del sistema en la retención de talento, la satisfacción de los empleados, y el alineamiento de habilidades con los objetivos organizacionales.
- **Reporte a la Dirección**: Presentar un reporte a la alta dirección con los logros alcanzados, las lecciones aprendidas y las oportunidades de mejora futuras.
""")

# Recursos Necesarios
st.header("Recursos Necesarios")
st.write("""
- **Equipo de Desarrollo**: Especialistas en IA, analistas de datos, desarrolladores de software, y diseñadores de UX/UI.
- **Colaboración Interdepartamental**: Involucramiento de Recursos Humanos, IT, y líderes de diferentes áreas de la organización.
- **Infraestructura Técnica**: Servidores, bases de datos, herramientas de análisis de datos, y plataformas de aprendizaje automático.
- **Capacitación**: Programas de formación para que los usuarios finales comprendan y se beneficien del sistema.
""")

# Cronograma Tentativo
st.header("Cronograma Tentativo")
st.write("""
- **Mes 1-2**: Análisis de requerimientos y definición de KPIs.
- **Mes 3-4**: Diseño del sistema y desarrollo del modelo de IA.
- **Mes 5-6**: Integración de datos, desarrollo de la interfaz de usuario, y entrenamiento del modelo.
- **Mes 7-8**: Implementación piloto, recopilación de feedback y ajustes.
- **Mes 9-10**: Despliegue completo y capacitación de usuarios.
- **Mes 11-12**: Monitoreo, evaluación y reporte final.
""")

# Beneficios Esperados
st.header("Beneficios Esperados")
st.write("""
- **Desarrollo Profesional Acelerado**: Los empleados recibirán planes personalizados que optimizan su crecimiento y alinean sus objetivos con los de la organización.
- **Mayor Retención de Talento**: Empleados más satisfechos y comprometidos, al ver que su desarrollo es una prioridad para la organización.
- **Mejora en la Eficiencia Organizacional**: Alineación más efectiva entre las competencias de los empleados y las necesidades estratégicas de PENSIONISSSTE.
""")

st.write("Este proyecto no solo mejorará la gestión del talento en PENSIONISSSTE, sino que también fortalecerá la cultura organizacional y asegurará que el personal esté bien equipado para afrontar los desafíos futuros.")
