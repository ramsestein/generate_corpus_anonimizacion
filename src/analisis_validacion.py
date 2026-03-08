#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Análisis de Inter-Rater Reliability para Validaciones de Anonimización Médica

Este script calcula métricas estandarizadas de acuerdo inter-evaluador
utilizando técnicas validadas en la literatura científica.

Referencias principales:
- Cohen, J. (1960). A coefficient of agreement for nominal scales.
- Landis, J. R., & Koch, G. G. (1977). The measurement of observer agreement.
- Fleiss, J. L. (1971). Measuring nominal scale agreement among many raters.
- Krippendorff, K. (2004). Reliability in content analysis.
- Gwet, K. L. (2008). Computing inter-rater reliability.
- Shrout, P. E., & Fleiss, J. L. (1979). Intraclass correlations.
"""

import pandas as pd
import numpy as np
import json
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import argparse

# Análisis estadístico
from sklearn.metrics import cohen_kappa_score, confusion_matrix, classification_report
from scipy import stats
from scipy.stats import chi2_contingency, chi2
from scipy.stats import bootstrap

# Visualizaciones
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Intentar importar librerías opcionales
try:
    from statsmodels.stats.inter_rater import fleiss_kappa
    FLEISS_AVAILABLE = True
except ImportError:
    FLEISS_AVAILABLE = False
    print("Advertencia: statsmodels no disponible. Fleiss' Kappa se calculará manualmente.")

try:
    import krippendorff
    KRIPPENDORFF_AVAILABLE = True
except ImportError:
    KRIPPENDORFF_AVAILABLE = False
    print("Advertencia: krippendorff no disponible. Se calculará manualmente.")

try:
    from statsmodels.stats.contingency_tables import mcnemar, cochrans_q
    CONTINGENCY_TABLES_AVAILABLE = True
except ImportError:
    CONTINGENCY_TABLES_AVAILABLE = False
    print("Advertencia: statsmodels.stats.contingency_tables no disponible.")

try:
    import pingouin as pg
    PINGOUIN_AVAILABLE = True
except ImportError:
    PINGOUIN_AVAILABLE = False
    print("Advertencia: pingouin no disponible. Se usarán implementaciones alternativas.")


class InterRaterAnalysis:
    """
    Clase principal para análisis de inter-rater reliability.
    
    Calcula métricas estándar de acuerdo entre evaluadores múltiples.
    """
    
    def __init__(self, csv_path: str, output_dir: str = "./resultados_validacion"):
        """
        Inicializa el analizador.
        
        Args:
            csv_path: Ruta al archivo CSV con validaciones
            output_dir: Directorio para guardar resultados
        """
        self.csv_path = csv_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        (self.output_dir / 'visualizaciones').mkdir(exist_ok=True)
        
        self.df = None
        self.evaluadores = []
        self.resultados_completos = {}
        
        self._cargar_datos()
    
    def _cargar_datos(self):
        """Carga y preprocesa los datos del CSV."""
        print("Cargando datos...")
        
        # Intentar diferentes encodings
        for encoding in ['utf-8', 'latin-1', 'cp1252']:
            try:
                self.df = pd.read_csv(self.csv_path, sep=';', encoding=encoding)
                print(f"Datos cargados con encoding: {encoding}")
                break
            except Exception as e:
                if encoding == 'cp1252':
                    raise Exception(f"No se pudo cargar el archivo con ningún encoding: {e}")
                continue
        
        # Identificar evaluadores
        columnas = self.df.columns.tolist()
        self.evaluadores = []
        for col in columnas:
            if '_Resultado' in col:
                evaluador = col.replace('_Resultado', '')
                self.evaluadores.append(evaluador)
        
        print(f"Evaluadores identificados: {', '.join(self.evaluadores)}")
        print(f"Total de documentos: {len(self.df)}")
    
    def normalizar_resultados(self, valor: str) -> Optional[str]:
        """Normaliza valores de resultado a Si/No."""
        if pd.isna(valor):
            return None
        valor = str(valor).strip().lower()
        if valor in ['si', 'sí', 's', 'yes', 'y', '1', 'true']:
            return 'Si'
        elif valor in ['no', 'n', '0', 'false']:
            return 'No'
        return None
    
    def calcular_cohen_kappa(self, evaluador1: str, evaluador2: str) -> Dict:
        """
        Calcula Cohen's Kappa entre dos evaluadores.
        
        Referencia: Cohen, J. (1960). A coefficient of agreement for nominal scales.
        Interpretación: Landis & Koch (1977)
        
        Returns:
            Dict con kappa, interpretación, IC 95%, p-value
        """
        col1 = f"{evaluador1}_Resultado"
        col2 = f"{evaluador2}_Resultado"
        
        # Filtrar documentos evaluados por ambos
        mask = self.df[[col1, col2]].notna().all(axis=1)
        datos1 = self.df.loc[mask, col1].apply(self.normalizar_resultados)
        datos2 = self.df.loc[mask, col2].apply(self.normalizar_resultados)
        
        # Filtrar valores válidos
        mask_valido = datos1.notna() & datos2.notna()
        datos1 = datos1[mask_valido]
        datos2 = datos2[mask_valido]
        
        if len(datos1) < 2:
            return {
                'kappa': None,
                'interpretacion': 'Datos insuficientes',
                'ic_95': None,
                'p_value': None,
                'n': len(datos1)
            }
        
        # Calcular Kappa
        kappa = cohen_kappa_score(datos1, datos2)
        
        # Manejar NaN (puede ocurrir cuando no hay variación)
        if np.isnan(kappa):
            kappa = 0.0
        
        # Interpretación según Landis & Koch (1977)
        if kappa < 0:
            interp = "No acuerdo"
        elif kappa <= 0.20:
            interp = "Leve"
        elif kappa <= 0.40:
            interp = "Bajo"
        elif kappa <= 0.60:
            interp = "Moderado"
        elif kappa <= 0.80:
            interp = "Sustancial"
        else:
            interp = "Casi perfecto"
        
        # Calcular IC 95% usando método aproximado
        n = len(datos1)
        try:
            if abs(kappa - 1.0) < 1e-10:
                # Si kappa es 1.0, el IC es [1.0, 1.0]
                ic_inf = 1.0
                ic_sup = 1.0
            else:
                # Método aproximado usando error estándar
                var_kappa = (kappa * (1 - kappa)) / (n * (1 - kappa**2 + 1e-10))
                se = np.sqrt(max(0, var_kappa))
                ic_inf = max(-1, kappa - 1.96 * se)
                ic_sup = min(1, kappa + 1.96 * se)
        except:
            # Fallback: usar aproximación simple
            ic_inf = max(-1, kappa - 0.1)
            ic_sup = min(1, kappa + 0.1)
        
        # Test de significancia (aproximado)
        se_null = np.sqrt(1 / n) if n > 0 else 0
        z = kappa / se_null if se_null > 0 else 0
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))
        
        return {
            'kappa': float(kappa),
            'interpretacion': interp,
            'ic_95': [float(ic_inf), float(ic_sup)],
            'p_value': float(p_value),
            'n': int(n)
        }
    
    def calcular_fleiss_kappa(self) -> Dict:
        """
        Calcula Fleiss' Kappa para múltiples evaluadores.
        
        Referencia: Fleiss, J. L. (1971). Measuring nominal scale agreement among many raters.
        """
        # Preparar matriz: documentos x evaluadores
        matriz = []
        documentos_validos = []
        
        for idx, row in self.df.iterrows():
            evaluaciones = []
            for eval in self.evaluadores:
                col = f"{eval}_Resultado"
                valor = self.normalizar_resultados(row.get(col))
                if valor is not None:
                    evaluaciones.append(1 if valor == 'Si' else 0)
                else:
                    evaluaciones.append(np.nan)
            
            # Solo incluir documentos con al menos 2 evaluaciones
            if sum(~pd.isna(evaluaciones)) >= 2:
                matriz.append(evaluaciones)
                documentos_validos.append(idx)
        
        if len(matriz) < 2:
            return {'kappa': None, 'interpretacion': 'Datos insuficientes', 'n': 0}
        
        matriz = np.array(matriz)
        
        # Implementación manual de Fleiss' Kappa
        n = len(matriz)  # Número de documentos
        N = np.sum(~np.isnan(matriz))  # Total de evaluaciones
        
        # Calcular proporción de acuerdos
        p_j = []  # Proporción de evaluadores que asignaron categoría j
        for j in [0, 1]:  # No, Si
            p_j.append(np.sum(matriz == j) / N)
        
        # Acuerdo esperado
        P_e = sum(p**2 for p in p_j)
        
        # Acuerdo observado
        P_o = 0
        for i in range(n):
            n_i = np.sum(~np.isnan(matriz[i, :]))
            if n_i > 1:
                # Contar acuerdos por documento
                n_ij = [np.sum(matriz[i, :] == j) for j in [0, 1]]
                P_o_i = sum(n_ij[j] * (n_ij[j] - 1) for j in [0, 1]) / (n_i * (n_i - 1))
                P_o += P_o_i
        
        P_o = P_o / n
        
        # Kappa
        if P_e == 1:
            kappa = 1.0
        else:
            kappa = (P_o - P_e) / (1 - P_e)
        
        # Interpretación
        if kappa < 0:
            interp = "No acuerdo"
        elif kappa <= 0.20:
            interp = "Leve"
        elif kappa <= 0.40:
            interp = "Bajo"
        elif kappa <= 0.60:
            interp = "Moderado"
        elif kappa <= 0.80:
            interp = "Sustancial"
        else:
            interp = "Casi perfecto"
        
        return {
            'kappa': float(kappa),
            'interpretacion': interp,
            'n': int(n),
            'P_o': float(P_o),
            'P_e': float(P_e)
        }
    
    def calcular_krippendorff_alpha(self) -> Dict:
        """
        Calcula Krippendorff's Alpha.
        
        Referencia: Krippendorff, K. (2004). Reliability in content analysis.
        """
        # Preparar datos en formato requerido
        datos_codificados = []
        
        for idx, row in self.df.iterrows():
            evaluaciones = []
            for eval in self.evaluadores:
                col = f"{eval}_Resultado"
                valor = self.normalizar_resultados(row.get(col))
                if valor is not None:
                    evaluaciones.append(1 if valor == 'Si' else 0)
                else:
                    evaluaciones.append(None)
            
            # Solo documentos con al menos 2 evaluaciones
            if sum(e is not None for e in evaluaciones) >= 2:
                datos_codificados.append(evaluaciones)
        
        if len(datos_codificados) < 2:
            return {'alpha': None, 'interpretacion': 'Datos insuficientes', 'n': 0}
        
        # Convertir a formato para cálculo manual
        n_items = len(datos_codificados)
        n_raters = len(self.evaluadores)
        
        # Matriz de datos
        matriz = np.array(datos_codificados)
        
        # Calcular Alpha manualmente
        # Alpha = 1 - (D_o / D_e)
        # donde D_o es desacuerdo observado y D_e es desacuerdo esperado
        
        D_o = 0
        D_e = 0
        
        for item in range(n_items):
            valores = matriz[item, :]
            valores_validos = [v for v in valores if v is not None]
            
            if len(valores_validos) < 2:
                continue
            
            # Desacuerdo observado
            for i in range(len(valores_validos)):
                for j in range(i+1, len(valores_validos)):
                    if valores_validos[i] != valores_validos[j]:
                        D_o += 1
        
        # Desacuerdo esperado (asumiendo distribución uniforme)
        n_comparaciones = 0
        for item in range(n_items):
            valores = matriz[item, :]
            n_validos = sum(1 for v in valores if v is not None)
            if n_validos >= 2:
                n_comparaciones += n_validos * (n_validos - 1) / 2
        
        if n_comparaciones > 0:
            D_e = n_comparaciones * 0.5  # Probabilidad de desacuerdo = 0.5 para 2 categorías
        
        if D_e == 0:
            alpha = 1.0
        else:
            alpha = 1 - (D_o / D_e)
        
        # Interpretación
        if alpha >= 0.800:
            interp = "Conclusiones confiables"
        elif alpha >= 0.667:
            interp = "Conclusiones tentativas"
        else:
            interp = "Datos descartables"
        
        return {
            'alpha': float(alpha),
            'interpretacion': interp,
            'n': int(n_items),
            'D_o': float(D_o),
            'D_e': float(D_e)
        }
    
    def calcular_porcentaje_acuerdo(self) -> Dict:
        """Calcula porcentaje de acuerdo simple entre evaluadores."""
        acuerdos_totales = 0
        comparaciones_totales = 0
        
        # Por cada par de evaluadores
        for i, eval1 in enumerate(self.evaluadores):
            for eval2 in self.evaluadores[i+1:]:
                col1 = f"{eval1}_Resultado"
                col2 = f"{eval2}_Resultado"
                
                mask = self.df[[col1, col2]].notna().all(axis=1)
                datos1 = self.df.loc[mask, col1].apply(self.normalizar_resultados)
                datos2 = self.df.loc[mask, col2].apply(self.normalizar_resultados)
                
                mask_valido = datos1.notna() & datos2.notna()
                datos1 = datos1[mask_valido]
                datos2 = datos2[mask_valido]
                
                if len(datos1) > 0:
                    acuerdos = (datos1 == datos2).sum()
                    acuerdos_totales += acuerdos
                    comparaciones_totales += len(datos1)
        
        if comparaciones_totales == 0:
            return {'porcentaje': None, 'n': 0}
        
        porcentaje = (acuerdos_totales / comparaciones_totales) * 100
        
        # IC 95% usando Wilson Score
        z = 1.96
        n = comparaciones_totales
        p = acuerdos_totales / n
        denominador = 1 + (z**2 / n)
        centro = (p + (z**2 / (2*n))) / denominador
        margen = (z / denominador) * np.sqrt((p * (1-p) / n) + (z**2 / (4*n**2)))
        
        ic_inf = max(0, (centro - margen) * 100)
        ic_sup = min(100, (centro + margen) * 100)
        
        return {
            'porcentaje': float(porcentaje),
            'ic_95': [float(ic_inf), float(ic_sup)],
            'acuerdos': int(acuerdos_totales),
            'total_comparaciones': int(comparaciones_totales)
        }
    
    def calcular_icc(self) -> Dict:
        """
        Calcula Intraclass Correlation Coefficient.
        
        Referencia: Shrout & Fleiss (1979). Intraclass correlations.
        """
        if not PINGOUIN_AVAILABLE:
            # Implementación simplificada
            return {'icc': None, 'interpretacion': 'Pingouin no disponible'}
        
        # Preparar datos en formato largo
        datos_largo = []
        for idx, row in self.df.iterrows():
            for eval in self.evaluadores:
                col = f"{eval}_Resultado"
                valor = self.normalizar_resultados(row.get(col))
                if valor is not None:
                    datos_largo.append({
                        'documento': idx,
                        'evaluador': eval,
                        'resultado': 1 if valor == 'Si' else 0
                    })
        
        if len(datos_largo) < 10:
            return {'icc': None, 'interpretacion': 'Datos insuficientes', 'n': 0}
        
        df_largo = pd.DataFrame(datos_largo)
        
        try:
            icc = pg.intraclass_corr(data=df_largo, targets='documento', 
                                    raters='evaluador', ratings='resultado')
            
            icc_2_1 = icc[icc['Type'] == 'ICC2']['ICC'].values[0]
            
            # Interpretación
            if icc_2_1 < 0.50:
                interp = "Pobre"
            elif icc_2_1 < 0.75:
                interp = "Moderado"
            elif icc_2_1 < 0.90:
                interp = "Bueno"
            else:
                interp = "Excelente"
            
            return {
                'icc_2_1': float(icc_2_1),
                'interpretacion': interp,
                'n': len(df_largo)
            }
        except Exception as e:
            return {'icc': None, 'error': str(e)}
    
    def calcular_cronbach_alpha(self) -> Dict:
        """
        Calcula Cronbach's Alpha.
        
        Referencia: Cronbach, L. J. (1951). Coefficient alpha and the internal structure of tests.
        """
        if not PINGOUIN_AVAILABLE:
            return {'alpha': None, 'interpretacion': 'Pingouin no disponible'}
        
        # Preparar matriz evaluadores x documentos
        matriz = []
        for eval in self.evaluadores:
            col = f"{eval}_Resultado"
            valores = self.df[col].apply(self.normalizar_resultados)
            valores_codificados = valores.apply(lambda x: 1 if x == 'Si' else (0 if x == 'No' else np.nan))
            matriz.append(valores_codificados.values)
        
        matriz = np.array(matriz).T
        
        try:
            alpha = pg.cronbach_alpha(data=matriz)
            
            # Interpretación
            if alpha[0] < 0.70:
                interp = "Baja consistencia"
            elif alpha[0] < 0.80:
                interp = "Consistencia aceptable"
            elif alpha[0] < 0.90:
                interp = "Buena consistencia"
            else:
                interp = "Excelente consistencia"
            
            return {
                'alpha': float(alpha[0]),
                'interpretacion': interp,
                'ic_95': [float(alpha[1][0]), float(alpha[1][1])]
            }
        except Exception as e:
            return {'alpha': None, 'error': str(e)}
    
    def calcular_tasas_anonimizacion(self) -> Dict:
        """Calcula tasas de éxito, error y ambigüedad del proceso de anonimización."""
        total = len(self.df)
        
        # Consenso "Si"
        n_si = (self.df['Consenso'] == 'Si').sum()
        tasa_si = (n_si / total) * 100
        
        # Consenso "No"
        n_no = (self.df['Consenso'] == 'No').sum()
        tasa_no = (n_no / total) * 100
        
        # Sin consenso (Mixto)
        n_mixto = (self.df['Consenso'] == 'Mixto').sum()
        tasa_mixto = (n_mixto / total) * 100
        
        # IC 95% para tasa de éxito (Wilson Score)
        z = 1.96
        p = n_si / total
        denominador = 1 + (z**2 / total)
        centro = (p + (z**2 / (2*total))) / denominador
        margen = (z / denominador) * np.sqrt((p * (1-p) / total) + (z**2 / (4*total**2)))
        ic_inf = max(0, (centro - margen) * 100)
        ic_sup = min(100, (centro + margen) * 100)
        
        return {
            'tasa_exito': float(tasa_si),
            'tasa_error': float(tasa_no),
            'tasa_ambiguedad': float(tasa_mixto),
            'ic_95_exito': [float(ic_inf), float(ic_sup)],
            'n_si': int(n_si),
            'n_no': int(n_no),
            'n_mixto': int(n_mixto),
            'total': int(total)
        }
    
    def calcular_matriz_confusion_agregada(self) -> pd.DataFrame:
        """Calcula matriz de confusión agregada entre todos los pares de evaluadores."""
        todas_matrices = []
        
        for i, eval1 in enumerate(self.evaluadores):
            for eval2 in self.evaluadores[i+1:]:
                col1 = f"{eval1}_Resultado"
                col2 = f"{eval2}_Resultado"
                
                mask = self.df[[col1, col2]].notna().all(axis=1)
                datos1 = self.df.loc[mask, col1].apply(self.normalizar_resultados)
                datos2 = self.df.loc[mask, col2].apply(self.normalizar_resultados)
                
                mask_valido = datos1.notna() & datos2.notna()
                datos1 = datos1[mask_valido]
                datos2 = datos2[mask_valido]
                
                if len(datos1) > 0:
                    cm = confusion_matrix(datos1, datos2, labels=['No', 'Si'])
                    todas_matrices.append(cm)
        
        if len(todas_matrices) == 0:
            return pd.DataFrame()
        
        matriz_agregada = np.sum(todas_matrices, axis=0)
        
        df_cm = pd.DataFrame(matriz_agregada, 
                            index=['No (Real)', 'Si (Real)'],
                            columns=['No (Pred)', 'Si (Pred)'])
        
        return df_cm
    
    def calcular_chi_cuadrado(self, evaluador1: str, evaluador2: str) -> Dict:
        """Calcula test de Chi-cuadrado de Pearson para independencia."""
        col1 = f"{evaluador1}_Resultado"
        col2 = f"{evaluador2}_Resultado"
        
        mask = self.df[[col1, col2]].notna().all(axis=1)
        datos1 = self.df.loc[mask, col1].apply(self.normalizar_resultados)
        datos2 = self.df.loc[mask, col2].apply(self.normalizar_resultados)
        
        mask_valido = datos1.notna() & datos2.notna()
        datos1 = datos1[mask_valido]
        datos2 = datos2[mask_valido]
        
        if len(datos1) < 2:
            return {'chi2': None, 'p_value': None, 'n': 0}
        
        # Tabla de contingencia
        tabla = pd.crosstab(datos1, datos2)
        
        chi2_stat, p_value, dof, expected = chi2_contingency(tabla)
        
        return {
            'chi2': float(chi2_stat),
            'p_value': float(p_value),
            'grados_libertad': int(dof),
            'n': int(len(datos1))
        }
    
    def calcular_entropia_shannon(self) -> Dict:
        """Calcula índice de entropía de Shannon para medir incertidumbre."""
        distribuciones = {}
        
        for eval in self.evaluadores:
            col = f"{eval}_Resultado"
            valores = self.df[col].apply(self.normalizar_resultados)
            valores_validos = valores.dropna()
            
            if len(valores_validos) > 0:
                conteo = valores_validos.value_counts()
                prob = conteo / len(valores_validos)
                
                # Entropía
                entropia = -sum(p * np.log2(p) for p in prob if p > 0)
                distribuciones[eval] = {
                    'entropia': float(entropia),
                    'n': int(len(valores_validos))
                }
        
        return distribuciones
    
    def calcular_todas_metricas(self):
        """Calcula todas las métricas disponibles."""
        print("\n" + "="*60)
        print("CALCULANDO MÉTRICAS DE INTER-RATER RELIABILITY")
        print("="*60)
        
        resultados = {}
        
        # 1. Cohen's Kappa entre pares
        print("\n1. Calculando Cohen's Kappa entre pares de evaluadores...")
        kappas_pares = {}
        for i, eval1 in enumerate(self.evaluadores):
            for eval2 in self.evaluadores[i+1:]:
                print(f"   {eval1} vs {eval2}...", end=' ')
                kappa = self.calcular_cohen_kappa(eval1, eval2)
                kappas_pares[f"{eval1}_vs_{eval2}"] = kappa
                if kappa['kappa'] is not None and not np.isnan(kappa['kappa']):
                    print(f"Kappa = {kappa['kappa']:.3f} ({kappa['interpretacion']})")
                else:
                    print("Datos insuficientes o sin variación")
        resultados['cohen_kappa_pares'] = kappas_pares
        
        # 2. Fleiss' Kappa
        print("\n2. Calculando Fleiss' Kappa...")
        fleiss = self.calcular_fleiss_kappa()
        resultados['fleiss_kappa'] = fleiss
        if fleiss['kappa'] is not None:
            print(f"   Fleiss' Kappa = {fleiss['kappa']:.3f} ({fleiss['interpretacion']})")
        
        # 3. Krippendorff's Alpha
        print("\n3. Calculando Krippendorff's Alpha...")
        kripp = self.calcular_krippendorff_alpha()
        resultados['krippendorff_alpha'] = kripp
        if kripp['alpha'] is not None:
            print(f"   Krippendorff's Alpha = {kripp['alpha']:.3f} ({kripp['interpretacion']})")
        
        # 4. Porcentaje de acuerdo
        print("\n4. Calculando porcentaje de acuerdo...")
        acuerdo = self.calcular_porcentaje_acuerdo()
        resultados['porcentaje_acuerdo'] = acuerdo
        if acuerdo['porcentaje'] is not None:
            print(f"   Porcentaje de acuerdo = {acuerdo['porcentaje']:.2f}%")
        
        # 5. ICC
        print("\n5. Calculando ICC...")
        icc = self.calcular_icc()
        resultados['icc'] = icc
        if icc.get('icc_2_1') is not None:
            print(f"   ICC(2,1) = {icc['icc_2_1']:.3f} ({icc['interpretacion']})")
        
        # 6. Cronbach's Alpha
        print("\n6. Calculando Cronbach's Alpha...")
        cronbach = self.calcular_cronbach_alpha()
        resultados['cronbach_alpha'] = cronbach
        if cronbach.get('alpha') is not None:
            print(f"   Cronbach's Alpha = {cronbach['alpha']:.3f} ({cronbach['interpretacion']})")
        
        # 7. Tasas de anonimización
        print("\n7. Calculando tasas de anonimización...")
        tasas = self.calcular_tasas_anonimizacion()
        resultados['tasas_anonimizacion'] = tasas
        print(f"   Tasa de éxito: {tasas['tasa_exito']:.2f}%")
        print(f"   Tasa de error: {tasas['tasa_error']:.2f}%")
        print(f"   Tasa de ambigüedad: {tasas['tasa_ambiguedad']:.2f}%")
        
        # 8. Entropía de Shannon
        print("\n8. Calculando entropía de Shannon...")
        entropia = self.calcular_entropia_shannon()
        resultados['entropia_shannon'] = entropia
        
        # 9. Matriz de confusión agregada
        print("\n9. Calculando matriz de confusión agregada...")
        matriz_conf = self.calcular_matriz_confusion_agregada()
        resultados['matriz_confusion'] = matriz_conf.to_dict() if not matriz_conf.empty else {}
        
        self.resultados_completos = resultados
        print("\n" + "="*60)
        print("CÁLCULO COMPLETADO")
        print("="*60)
        
        return resultados
    
    def generar_visualizaciones(self):
        """Genera todas las visualizaciones."""
        print("\nGenerando visualizaciones...")
        
        # 1. Heatmap de Cohen's Kappa
        if 'cohen_kappa_pares' in self.resultados_completos:
            self._plot_heatmap_kappa()
        
        # 2. Distribución de consenso
        self._plot_distribucion_consenso()
        
        # 3. Variabilidad por evaluador
        self._plot_variabilidad_evaluadores()
        
        # 4. Matriz de confusión
        if 'matriz_confusion' in self.resultados_completos:
            self._plot_matriz_confusion()
        
        print("Visualizaciones guardadas en:", self.output_dir / 'visualizaciones')
    
    def _plot_heatmap_kappa(self):
        """Genera heatmap de Cohen's Kappa entre evaluadores."""
        kappas = self.resultados_completos['cohen_kappa_pares']
        
        # Crear matriz
        matriz = np.ones((len(self.evaluadores), len(self.evaluadores))) * np.nan
        for i, eval1 in enumerate(self.evaluadores):
            for j, eval2 in enumerate(self.evaluadores):
                if i == j:
                    matriz[i, j] = 1.0
                else:
                    key = f"{eval1}_vs_{eval2}" if i < j else f"{eval2}_vs_{eval1}"
                    if key in kappas and kappas[key]['kappa'] is not None:
                        matriz[i, j] = kappas[key]['kappa']
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(matriz, annot=True, fmt='.3f', cmap='RdYlGn', 
                   vmin=-1, vmax=1, center=0,
                   xticklabels=self.evaluadores,
                   yticklabels=self.evaluadores,
                   cbar_kws={'label': "Cohen's Kappa"})
        plt.title("Cohen's Kappa entre Evaluadores", fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'visualizaciones' / 'heatmap_kappa.png', dpi=300)
        plt.close()
    
    def _plot_distribucion_consenso(self):
        """Gráfico de barras con distribución de consenso."""
        consenso_counts = self.df['Consenso'].value_counts()
        
        plt.figure(figsize=(8, 6))
        consenso_counts.plot(kind='bar', color=['green', 'red', 'orange'])
        plt.title('Distribución de Consenso en Evaluaciones', fontsize=14, fontweight='bold')
        plt.xlabel('Consenso')
        plt.ylabel('Número de Documentos')
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'visualizaciones' / 'distribucion_consenso.png', dpi=300)
        plt.close()
    
    def _plot_variabilidad_evaluadores(self):
        """Box plot de variabilidad por evaluador."""
        datos_box = []
        for eval in self.evaluadores:
            col = f"{eval}_Resultado"
            valores = self.df[col].apply(self.normalizar_resultados)
            valores_codificados = valores.apply(lambda x: 1 if x == 'Si' else (0 if x == 'No' else np.nan))
            datos_box.append(valores_codificados.dropna().values)
        
        plt.figure(figsize=(10, 6))
        plt.boxplot(datos_box, labels=self.evaluadores)
        plt.title('Distribución de Evaluaciones por Evaluador', fontsize=14, fontweight='bold')
        plt.ylabel('Resultado (0=No, 1=Si)')
        plt.xlabel('Evaluador')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'visualizaciones' / 'variabilidad_evaluadores.png', dpi=300)
        plt.close()
    
    def _plot_matriz_confusion(self):
        """Visualiza matriz de confusión agregada."""
        if 'matriz_confusion' not in self.resultados_completos:
            return
        
        matriz = self.resultados_completos['matriz_confusion']
        if not matriz:
            return
        
        df_cm = pd.DataFrame(matriz)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Matriz de Confusión Agregada', fontsize=14, fontweight='bold')
        plt.ylabel('Resultado Real')
        plt.xlabel('Resultado Predicho')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'visualizaciones' / 'matriz_confusion.png', dpi=300)
        plt.close()
    
    def generar_reporte_markdown(self):
        """Genera reporte en formato Markdown."""
        reporte = []
        
        reporte.append("# Reporte de Inter-Rater Reliability\n")
        reporte.append("## Análisis de Validaciones de Anonimización Médica\n")
        reporte.append(f"**Fecha de análisis:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        reporte.append(f"**Total de documentos:** {len(self.df)}\n")
        reporte.append(f"**Evaluadores:** {', '.join(self.evaluadores)}\n")
        reporte.append("\n---\n")
        
        # Cohen's Kappa
        if 'cohen_kappa_pares' in self.resultados_completos:
            reporte.append("\n## 1. Cohen's Kappa\n")
            reporte.append("Acuerdo entre pares de evaluadores.\n\n")
            reporte.append("| Evaluadores | Kappa | Interpretación | IC 95% | p-value | N |\n")
            reporte.append("|-------------|-------|----------------|--------|---------|---|\n")
            
            for par, kappa in self.resultados_completos['cohen_kappa_pares'].items():
                if kappa['kappa'] is not None:
                    eval1, eval2 = par.split('_vs_')
                    ic_str = f"[{kappa['ic_95'][0]:.3f}, {kappa['ic_95'][1]:.3f}]" if kappa['ic_95'] else "N/A"
                    p_str = f"{kappa['p_value']:.4f}" if kappa['p_value'] else "N/A"
                    reporte.append(f"| {eval1} vs {eval2} | {kappa['kappa']:.3f} | {kappa['interpretacion']} | {ic_str} | {p_str} | {kappa['n']} |\n")
        
        # Fleiss' Kappa
        if 'fleiss_kappa' in self.resultados_completos:
            fleiss = self.resultados_completos['fleiss_kappa']
            if fleiss['kappa'] is not None:
                reporte.append("\n## 2. Fleiss' Kappa\n")
                reporte.append(f"**Kappa:** {fleiss['kappa']:.3f}\n")
                reporte.append(f"**Interpretación:** {fleiss['interpretacion']}\n")
                reporte.append(f"**Documentos analizados:** {fleiss['n']}\n")
        
        # Krippendorff's Alpha
        if 'krippendorff_alpha' in self.resultados_completos:
            kripp = self.resultados_completos['krippendorff_alpha']
            if kripp['alpha'] is not None:
                reporte.append("\n## 3. Krippendorff's Alpha\n")
                reporte.append(f"**Alpha:** {kripp['alpha']:.3f}\n")
                reporte.append(f"**Interpretación:** {kripp['interpretacion']}\n")
                reporte.append(f"**Documentos analizados:** {kripp['n']}\n")
        
        # Tasas de anonimización
        if 'tasas_anonimizacion' in self.resultados_completos:
            tasas = self.resultados_completos['tasas_anonimizacion']
            reporte.append("\n## 4. Tasas de Anonimización\n")
            reporte.append(f"**Tasa de éxito:** {tasas['tasa_exito']:.2f}% (IC 95%: [{tasas['ic_95_exito'][0]:.2f}, {tasas['ic_95_exito'][1]:.2f}])\n")
            reporte.append(f"**Tasa de error:** {tasas['tasa_error']:.2f}%\n")
            reporte.append(f"**Tasa de ambigüedad:** {tasas['tasa_ambiguedad']:.2f}%\n")
        
        # Referencias
        reporte.append("\n---\n")
        reporte.append("\n## Referencias Bibliográficas\n\n")
        reporte.append("1. Cohen, J. (1960). A coefficient of agreement for nominal scales. Educational and Psychological Measurement, 20(1), 37-46.\n")
        reporte.append("2. Landis, J. R., & Koch, G. G. (1977). The measurement of observer agreement for categorical data. Biometrics, 33(1), 159-174.\n")
        reporte.append("3. Fleiss, J. L. (1971). Measuring nominal scale agreement among many raters. Psychological Bulletin, 76(5), 378-382.\n")
        reporte.append("4. Krippendorff, K. (2004). Reliability in content analysis: Some common misconceptions and recommendations. Human Communication Research, 30(3), 411-433.\n")
        reporte.append("5. Shrout, P. E., & Fleiss, J. L. (1979). Intraclass correlations: Uses in assessing rater reliability. Psychological Bulletin, 86(2), 420-428.\n")
        
        # Guardar
        with open(self.output_dir / 'reporte_validacion.md', 'w', encoding='utf-8') as f:
            f.writelines(reporte)
        
        print(f"Reporte Markdown guardado en: {self.output_dir / 'reporte_validacion.md'}")
    
    def exportar_resultados(self):
        """Exporta todos los resultados a CSV y JSON."""
        # JSON completo
        with open(self.output_dir / 'resultados_completos.json', 'w', encoding='utf-8') as f:
            json.dump(self.resultados_completos, f, indent=2, ensure_ascii=False)
        
        # Tabla resumen de métricas principales
        filas = []
        
        # Cohen's Kappa
        if 'cohen_kappa_pares' in self.resultados_completos:
            for par, kappa in self.resultados_completos['cohen_kappa_pares'].items():
                if kappa['kappa'] is not None:
                    filas.append({
                        'Metrica': 'Cohen_Kappa',
                        'Evaluadores': par,
                        'Valor': kappa['kappa'],
                        'Interpretacion': kappa['interpretacion'],
                        'IC_95_inf': kappa['ic_95'][0] if kappa['ic_95'] else None,
                        'IC_95_sup': kappa['ic_95'][1] if kappa['ic_95'] else None,
                        'p_value': kappa['p_value'],
                        'N': kappa['n']
                    })
        
        # Otras métricas
        if 'fleiss_kappa' in self.resultados_completos and self.resultados_completos['fleiss_kappa']['kappa']:
            fleiss = self.resultados_completos['fleiss_kappa']
            filas.append({
                'Metrica': 'Fleiss_Kappa',
                'Evaluadores': 'Todos',
                'Valor': fleiss['kappa'],
                'Interpretacion': fleiss['interpretacion'],
                'N': fleiss['n']
            })
        
        if 'krippendorff_alpha' in self.resultados_completos and self.resultados_completos['krippendorff_alpha']['alpha']:
            kripp = self.resultados_completos['krippendorff_alpha']
            filas.append({
                'Metrica': 'Krippendorff_Alpha',
                'Evaluadores': 'Todos',
                'Valor': kripp['alpha'],
                'Interpretacion': kripp['interpretacion'],
                'N': kripp['n']
            })
        
        if filas:
            df_resultados = pd.DataFrame(filas)
            df_resultados.to_csv(self.output_dir / 'metricas_validacion.csv', 
                                index=False, encoding='utf-8-sig', sep=';')
            print(f"Resultados CSV guardados en: {self.output_dir / 'metricas_validacion.csv'}")


def main():
    """Función principal."""
    parser = argparse.ArgumentParser(description='Análisis de Inter-Rater Reliability')
    parser.add_argument('--input', '-i', 
                      default='results_human/tabla_consolidada_validaciones.csv',
                      help='Ruta al archivo CSV de entrada')
    parser.add_argument('--output', '-o',
                      default='./resultados_validacion',
                      help='Directorio de salida')
    
    args = parser.parse_args()
    
    # Crear analizador
    analyzer = InterRaterAnalysis(args.input, args.output)
    
    # Calcular todas las métricas
    analyzer.calcular_todas_metricas()
    
    # Generar visualizaciones
    analyzer.generar_visualizaciones()
    
    # Generar reportes
    analyzer.generar_reporte_markdown()
    
    # Exportar resultados
    analyzer.exportar_resultados()
    
    print("\n" + "="*60)
    print("ANÁLISIS COMPLETADO")
    print("="*60)
    print(f"Resultados guardados en: {analyzer.output_dir}")


if __name__ == '__main__':
    main()

