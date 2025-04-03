import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import pandas as pd
from datetime import datetime

# Configuración de la página
st.set_page_config(page_title="Calculadora de Newton-Raphson", layout="wide")
st.title("Calculadora Avanzada del Método de Newton-Raphson")

# Entrada de la función
st.sidebar.header("Parámetros de entrada")
funcion_str = st.sidebar.text_input("Función f(x):", "x**3 - 2*x - 5")  # Ejemplo por defecto

# Validación básica de la función ingresada
with st.sidebar.expander("Ayuda para ingresar funciones"):
    st.markdown("""
    Ejemplos de funciones válidas:
    - `x**2 - 4*x + 4` (una parábola)
    - `sin(x) - 0.5` (función trigonométrica)
    - `exp(x) - 3` (función exponencial)
    
    Operadores y funciones disponibles:
    - Operaciones básicas: `+`, `-`, `*`, `/`, `**` (potencia)
    - Funciones: `sin`, `cos`, `tan`, `exp`, `log`, `sqrt`
    """)

col1, col2 = st.sidebar.columns(2)
with col1:
    x0 = st.number_input("Valor inicial (x0):", value=3.0, step=0.1)
with col2:
    decimales = st.number_input("Decimales de precisión:", 
                              min_value=1, 
                              max_value=15, 
                              value=5,
                              help="Debe ser un entero entre 1 y 15")

# Parámetros adicionales con validación
max_iter = st.sidebar.number_input("Máximo de iteraciones:", 
                                  min_value=1,  # Ahora mínimo es 1
                                  max_value=1000, 
                                  value=100,
                                  help="Debe ser un entero positivo")
                                  
grafica_range = st.sidebar.slider("Rango de la gráfica (±):", min_value=1, max_value=20, value=5)

# Calcular tolerancia en base a los decimales
tolerancia = 10**-decimales

# Área para mostrar errores de validación
error_placeholder = st.sidebar.empty()



# Verificación previa de la función
valid_function = False
try:
    x = sp.symbols('x')
    f_expr = sp.sympify(funcion_str)
    
    # Verificar que no haya variables distintas de 'x'
    variables = list(f_expr.free_symbols)
    if any(str(var) != 'x' for var in variables):
        st.sidebar.error("⚠️ Solo se permite la variable 'x' en la función.")
    else:
        df_expr = sp.diff(f_expr, x)
        
        # Convertir a string para mostrar
        funcion_latex = sp.latex(f_expr)
        derivada_latex = sp.latex(df_expr)
        
        st.sidebar.latex(r"f(x) = " + funcion_latex)
        st.sidebar.latex(r"f'(x) = " + derivada_latex)
        valid_function = True
except sp.SympifyError:
    st.sidebar.error("⚠️ La función no es válida. Revise la sintaxis.")
except Exception as e:
    st.sidebar.error(f"⚠️ Error: {str(e)}")

# Botón para calcular
calcular_btn = st.sidebar.button("Calcular", type="primary", disabled=not valid_function)

# Función principal
if calcular_btn and valid_function:
    try:
        # Definir variable simbólica
        x = sp.symbols('x')

        # Convertir expresiones a funciones evaluables
        f = sp.lambdify(x, f_expr, "numpy")
        df = sp.lambdify(x, df_expr, "numpy")

        # Método de Newton-Raphson
        def newton_raphson(f, df, x0, tol, max_iter):
            iteraciones = []
            xn = float(x0)
            error = float('inf')
            i = 0
            convergencia = "No convergió"

            while error > tol and i < max_iter:
                try:
                    f_xn = f(xn)
                    df_xn = df(xn)
                except TypeError:
                    raise ValueError("La función no puede evaluarse numéricamente. Revise su definición.")

                if abs(df_xn) < 1e-14:  # Evita división por cero
                    convergencia = "Falló - Derivada nula"
                    break

                # Newton-Raphson estándar
                x_next = xn - f_xn / df_xn

                error = abs(x_next - xn)
                iteraciones.append((i, xn, f_xn, df_xn, x_next, error))
                
                if i > 3 and abs(iteraciones[i-2][1] - xn) < tol*10:
                    convergencia = "Oscilación detectada"
                    break
                    
                xn = x_next
                i += 1

            if i < max_iter and convergencia == "No convergió":
                convergencia = "Convergencia rápida" if i < 5 else "Convergencia normal"
            elif i >= max_iter:
                convergencia = "Máximo de iteraciones alcanzado"

            return iteraciones, convergencia

        # Calcular iteraciones
        with st.spinner("Calculando..."):
            iteraciones, convergencia = newton_raphson(f, df, x0, tolerancia, max_iter)

        # Mostrar resultados (igual que antes)
        col1, col2 = st.columns([3, 2])

        with col1:
            st.subheader("Resultados")
            
            if iteraciones:
                resultado = {
                    "Función": funcion_str,
                    "Derivada": str(sp.simplify(df_expr)),
                    "Valor inicial": f"{x0:.{decimales}f}",
                    "Iteraciones realizadas": len(iteraciones),
                    "Raíz encontrada": f"{iteraciones[-1][4]:.{decimales}f}",
                    "Valor de f(raíz)": f"{f(iteraciones[-1][4]):.{decimales}f}",
                    "Error final": f"{iteraciones[-1][5]:.{decimales}f}",
                    "Estado de convergencia": convergencia
                }
                
                for key, value in resultado.items():
                    st.write(f"**{key}:** {value}")
                
            
                st.subheader("Tabla de iteraciones")
                tabla_datos = [
                    {"Iter.": it[0] + 1, 
                     "x_n": f"{it[1]:.{decimales}f}", 
                     "f(x_n)": f"{it[2]:.{decimales}f}", 
                     "f'(x_n)": f"{it[3]:.{decimales}f}",
                     "x_n+1": f"{it[4]:.{decimales}f}",
                     "Error": f"{it[5]:.{decimales}f}"}
                    for it in iteraciones
                ]
                st.dataframe(pd.DataFrame(tabla_datos), use_container_width=True)
                
                df_download = pd.DataFrame(tabla_datos)
                csv = df_download.to_csv(index=False)
                
                st.download_button(
                    label="Descargar resultados como CSV",
                    data=csv,
                    file_name=f"newton_raphson_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            else:
                st.error("No se pudo calcular la raíz con los parámetros dados.")

        with col2:
            if iteraciones:
                st.subheader("Gráfica de la función")
                ultima_x = iteraciones[-1][4]
                x_vals = np.linspace(ultima_x - grafica_range, ultima_x + grafica_range, 500)
                y_vals = f(x_vals)
                
                fig, ax = plt.subplots(figsize=(6, 5))
                ax.plot(x_vals, y_vals, label=f"f(x)", color="blue")
                ax.axhline(0, color="black", linewidth=0.7, alpha=0.7)
                ax.axvline(ultima_x, color="green", linestyle="--", alpha=0.5, label=f"Raíz x={ultima_x:.{decimales}f}")
                ax.scatter([it[1] for it in iteraciones], [f(it[1]) for it in iteraciones], color="red", s=50, alpha=0.6, label="Iteraciones")
                
                for i in range(len(iteraciones)-1):
                    x1, x2 = iteraciones[i][1], iteraciones[i+1][1]
                    y1, y2 = f(x1), f(x2)
                    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                                arrowprops=dict(arrowstyle="->", lw=1.0, alpha=0.5, color="gray"))
                
                ax.set_xlabel("x")
                ax.set_ylabel("f(x)")
                ax.grid(alpha=0.3)
                ax.legend()
                st.pyplot(fig)
            
                st.subheader("Velocidad de convergencia")
                fig2, ax2 = plt.subplots(figsize=(6, 3))
                errores = [it[5] for it in iteraciones]
                ax2.semilogy(range(1, len(errores)+1), errores, 'o-', color="purple")
                ax2.set_xlabel("Iteración")
                ax2.set_ylabel("Error")
                ax2.grid(True, which="both", alpha=0.3)
                st.pyplot(fig2)
        # Mostrar el procedimiento detallado de cada iteración
        with st.expander("Ver procedimiento detallado de las iteraciones", expanded=False):
            for i, it in enumerate(iteraciones):
                st.markdown(f"**Iteración {i + 1}:**")
                
                formula = f"x_{{n+1}} = x_n - \\frac{{f(x_n)}}{{f'(x_n)}}"
                calculo = f"x_{{n+1}} = {it[1]:.{decimales}f} - \\frac{{{it[2]:.{decimales}f}}}{{{it[3]:.{decimales}f}}}"
                
                col1, col2 = st.columns(2)
                with col1:
                    st.latex(f"x_n = {it[1]:.{decimales}f}")
                    st.latex(f"f(x_n) = {it[2]:.{decimales}f}")
                    st.latex(f"f'(x_n) = {it[3]:.{decimales}f}")
                
                with col2:
                    st.latex(formula)
                    st.latex(calculo + f" = {it[4]:.{decimales}f}")
                    st.latex(f"\\text{{Error}} = |x_{{n+1}} - x_n| = |{it[4]:.{decimales}f} - {it[1]:.{decimales}f}| = {it[5]:.{decimales}f}")
                
                st.markdown("---")

    except ValueError as e:
        st.error(f"Error: {str(e)}")    
    except ZeroDivisionError:
        st.error("Error: División por cero durante el cálculo. Intente con otro valor inicial.")
    except Exception as e:
        st.error(f"Error inesperado: {str(e)}")


else:
    # Sección de información del equipo en el sidebar
    with st.sidebar.expander("📝 Información del equipo", expanded=True):
        st.markdown("""
        **Universidad:** Universidad Tecnológica de Tula-Tepeji  
        **Carrera:** Ing.Desarrollo de Software Multiplataforma  
        **Materia:** Matemáticas para ing. 2  
        **Profesor:** Rogelio Rivas Cano 
        
        **Integrantes del equipo:**
        1. Angel Leobardo Perez Perez 
        2. Orlando Galvan Vargas 
        3. Brian Emmanuel Flores Hernandez 
        
        **Fecha de entrega:** 03/04/2025
        """)
    
    # Mostrar información inicial cuando no se ha calculado nada
    st.info("""
    Esta aplicación implementa el método de Newton-Raphson para encontrar raíces de funciones.
    
    ### Para comenzar:
    1. Ingrese una función en términos de 'x' en el panel izquierdo
    2. Establezca un valor inicial (x0)
    3. Ajuste la precisión deseada
    4. Haga clic en "Calcular"
    """)
    
    # Mostrar gráfica de una función ejemplo
    with st.expander("Ver ejemplo de gráfica", expanded=True):
        st.write("Gráfica de la función de ejemplo: f(x) = x³ - 2x - 5")
        
        x_sample = np.linspace(-4, 4, 100)
        y_sample = x_sample**3 - 2*x_sample - 5
        
        fig, ax = plt.subplots()
        ax.plot(x_sample, y_sample, color="blue")
        ax.axhline(0, color="black", linewidth=0.7)
        ax.set_xlabel("x")
        ax.set_ylabel("f(x)")
        ax.grid(alpha=0.3)
        st.pyplot(fig)