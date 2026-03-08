#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Análisis completo de detección de contenido generado por IA
utilizando métricas estandarizadas reportadas en literatura científica.
"""

import os
import re
import json
import math
import statistics
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Tuple
import numpy as np
from tqdm import tqdm

try:
    import nltk
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import SnowballStemmer
    from nltk.util import ngrams
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
except ImportError:
    print("Instalando nltk...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'nltk'])
    import nltk
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.corpus import stopwords
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)

try:
    import textstat
except ImportError:
    print("Instalando textstat...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'textstat'])
    import textstat

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    print("Advertencia: sentence-transformers no disponible. Algunas métricas de coherencia no se calcularán.")
    SentenceTransformer = None

try:
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    import torch
except ImportError:
    print("Advertencia: transformers no disponible. Perplejidad aproximada será usada.")
    GPT2LMHeadModel = None

class DetectorIA:
    def __init__(self):
        """Inicializa el detector con modelos necesarios."""
        self.stemmer = SnowballStemmer('spanish')
        self.spanish_stopwords = set(stopwords.words('spanish'))
        
        # Cargar modelo de embeddings si está disponible
        self.embedding_model = None
        if SentenceTransformer:
            try:
                self.embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            except:
                print("No se pudo cargar modelo de embeddings. Usando métricas alternativas.")
        
        # Cargar modelo de lenguaje para perplejidad si está disponible
        self.lm_model = None
        self.lm_tokenizer = None
        if GPT2LMHeadModel and torch.cuda.is_available() or True:  # Intentar siempre
            try:
                # Usar modelo pequeño para eficiencia
                self.lm_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
                self.lm_model = GPT2LMHeadModel.from_pretrained('gpt2')
                self.lm_model.eval()
            except:
                print("No se pudo cargar modelo de lenguaje. Usando aproximación de perplejidad.")
    
    def preprocesar_texto(self, texto: str) -> Tuple[List[str], List[str], List[str]]:
        """Preprocesa el texto y retorna tokens, oraciones y palabras."""
        # Normalizar
        texto = texto.replace('\n', ' ').replace('\r', ' ')
        texto = re.sub(r'\s+', ' ', texto)
        
        # Tokenizar
        try:
            oraciones = sent_tokenize(texto, language='spanish')
        except:
            oraciones = [s.strip() for s in texto.split('.') if s.strip()]
        
        palabras = []
        for oracion in oraciones:
            try:
                tokens = word_tokenize(oracion.lower(), language='spanish')
                palabras.extend([t for t in tokens if t.isalnum()])
            except:
                palabras.extend([w.lower() for w in oracion.split() if w.isalnum()])
        
        return palabras, oraciones, texto.split()
    
    def calcular_perplejidad(self, texto: str) -> float:
        """Calcula perplejidad estimada del texto."""
        palabras, oraciones, _ = self.preprocesar_texto(texto)
        
        if len(palabras) < 2:
            return 0.0
        
        # Si hay modelo de lenguaje, usarlo
        if self.lm_model and self.lm_tokenizer:
            try:
                tokens = self.lm_tokenizer.encode(texto[:512], return_tensors='pt')
                with torch.no_grad():
                    outputs = self.lm_model(tokens, labels=tokens)
                    loss = outputs.loss
                    perplexity = torch.exp(loss).item()
                    return perplexity
            except:
                pass
        
        # Método aproximado: usar distribución de n-gramas
        bigrams = list(ngrams(palabras, 2))
        if len(bigrams) == 0:
            return 0.0
        
        bigram_counts = Counter(bigrams)
        total_bigrams = len(bigrams)
        
        # Calcular entropía
        entropy = 0.0
        for count in bigram_counts.values():
            prob = count / total_bigrams
            if prob > 0:
                entropy -= prob * math.log2(prob)
        
        perplexity = 2 ** entropy
        return perplexity
    
    def calcular_burstiness(self, texto: str) -> float:
        """Calcula burstiness: ratio de variación en longitud de frases."""
        palabras, oraciones, _ = self.preprocesar_texto(texto)
        
        if len(oraciones) < 2:
            return 0.0
        
        longitudes = [len(s.split()) for s in oraciones if s.strip()]
        if len(longitudes) == 0:
            return 0.0
        
        media = statistics.mean(longitudes)
        if media == 0:
            return 0.0
        
        desv_est = statistics.stdev(longitudes) if len(longitudes) > 1 else 0.0
        burstiness = desv_est / media if media > 0 else 0.0
        
        return burstiness
    
    def calcular_ttr(self, texto: str) -> Tuple[float, float]:
        """Calcula Type-Token Ratio estándar y Moving-average TTR."""
        palabras, _, _ = self.preprocesar_texto(texto)
        
        if len(palabras) == 0:
            return 0.0, 0.0
        
        tipos_unicos = len(set(palabras))
        ttr_estandar = tipos_unicos / len(palabras)
        
        # MTTR: promedio de TTR en ventanas de 50 palabras
        if len(palabras) < 50:
            mttr = ttr_estandar
        else:
            ventana = 50
            ttrs = []
            for i in range(0, len(palabras) - ventana + 1, ventana):
                ventana_palabras = palabras[i:i+ventana]
                tipos_ventana = len(set(ventana_palabras))
                ttrs.append(tipos_ventana / len(ventana_palabras))
            mttr = statistics.mean(ttrs) if ttrs else ttr_estandar
        
        return ttr_estandar, mttr
    
    def calcular_hapax_legomena(self, texto: str) -> float:
        """Calcula ratio de palabras que aparecen solo una vez."""
        palabras, _, _ = self.preprocesar_texto(texto)
        
        if len(palabras) == 0:
            return 0.0
        
        conteo = Counter(palabras)
        hapax = sum(1 for count in conteo.values() if count == 1)
        ratio = hapax / len(palabras)
        
        return ratio
    
    def calcular_yule_k(self, texto: str) -> float:
        """Calcula índice de diversidad léxica de Yule (K)."""
        palabras, _, _ = self.preprocesar_texto(texto)
        
        if len(palabras) == 0:
            return 0.0
        
        conteo = Counter(palabras)
        N = len(palabras)
        
        # M1 = N
        M1 = N
        
        # M2 = Σ V_i * i^2 donde V_i es número de palabras que aparecen i veces
        frecuencias = Counter(conteo.values())
        M2 = sum(freq * (count ** 2) for count, freq in frecuencias.items())
        
        if M1 == 0:
            return 0.0
        
        K = 10000 * (M2 - M1) / (M1 ** 2)
        return K
    
    def calcular_mls(self, texto: str) -> Tuple[float, float]:
        """Calcula longitud media de frases y coeficiente de variación."""
        palabras, oraciones, _ = self.preprocesar_texto(texto)
        
        if len(oraciones) == 0:
            return 0.0, 0.0
        
        longitudes = [len(s.split()) for s in oraciones if s.strip()]
        if len(longitudes) == 0:
            return 0.0, 0.0
        
        media = statistics.mean(longitudes)
        cv = statistics.stdev(longitudes) / media if media > 0 and len(longitudes) > 1 else 0.0
        
        return media, cv
    
    def calcular_flesch_kincaid(self, texto: str) -> Tuple[float, float]:
        """Calcula índice de Flesch-Kincaid y su varianza."""
        try:
            # Flesch Reading Ease
            score = textstat.flesch_reading_ease(texto)
            
            # Calcular varianza en ventanas
            oraciones = sent_tokenize(texto, language='spanish')
            if len(oraciones) < 2:
                return score, 0.0
            
            scores = []
            for oracion in oraciones:
                try:
                    s = textstat.flesch_reading_ease(oracion)
                    scores.append(s)
                except:
                    pass
            
            varianza = statistics.variance(scores) if len(scores) > 1 else 0.0
            return score, varianza
        except:
            return 0.0, 0.0
    
    def calcular_coherencia_semantica(self, texto: str) -> Tuple[float, float]:
        """Calcula coherencia semántica usando embeddings."""
        if not self.embedding_model:
            return 0.0, 0.0
        
        try:
            oraciones = sent_tokenize(texto, language='spanish')
            if len(oraciones) < 2:
                return 0.0, 0.0
            
            # Obtener embeddings
            embeddings = self.embedding_model.encode(oraciones)
            
            # Calcular similitudes entre pares consecutivos
            similitudes = []
            for i in range(len(embeddings) - 1):
                sim = cosine_similarity([embeddings[i]], [embeddings[i+1]])[0][0]
                similitudes.append(sim)
            
            if len(similitudes) == 0:
                return 0.0, 0.0
            
            media = statistics.mean(similitudes)
            varianza = statistics.variance(similitudes) if len(similitudes) > 1 else 0.0
            
            return media, varianza
        except:
            return 0.0, 0.0
    
    def calcular_ratio_funcion_contenido(self, texto: str) -> float:
        """Calcula ratio de palabras función vs contenido."""
        palabras, _, _ = self.preprocesar_texto(texto)
        
        if len(palabras) == 0:
            return 0.0
        
        # Palabras función son stopwords
        palabras_funcion = sum(1 for p in palabras if p in self.spanish_stopwords)
        ratio = palabras_funcion / len(palabras)
        
        return ratio
    
    def analizar_documento(self, texto: str) -> Dict:
        """Analiza un documento completo y retorna todas las métricas."""
        if not texto or len(texto.strip()) < 10:
            return None
        
        resultado = {}
        
        # 1. Perplejidad y Burstiness
        resultado['perplejidad'] = self.calcular_perplejidad(texto)
        resultado['burstiness'] = self.calcular_burstiness(texto)
        
        # 2. Métricas léxicas
        ttr, mttr = self.calcular_ttr(texto)
        resultado['ttr'] = ttr
        resultado['mttr'] = mttr
        resultado['hapax_ratio'] = self.calcular_hapax_legomena(texto)
        resultado['yule_k'] = self.calcular_yule_k(texto)
        
        # 3. Métricas sintácticas
        mls, cv_mls = self.calcular_mls(texto)
        resultado['mls'] = mls
        resultado['cv_mls'] = cv_mls
        fk_score, fk_var = self.calcular_flesch_kincaid(texto)
        resultado['flesch_kincaid'] = fk_score
        resultado['flesch_varianza'] = fk_var
        
        # 4. Coherencia
        coherencia, coherencia_var = self.calcular_coherencia_semantica(texto)
        resultado['coherencia_semantica'] = coherencia
        resultado['coherencia_varianza'] = coherencia_var
        
        # 5. Estilométricas
        resultado['ratio_funcion_contenido'] = self.calcular_ratio_funcion_contenido(texto)
        
        # Clasificación y probabilidad
        resultado['clasificacion'] = self.clasificar_documento(resultado)
        resultado['probabilidad_humano'] = self.calcular_probabilidad_humano(resultado)
        
        return resultado
    
    def clasificar_documento(self, metricas: Dict) -> str:
        """Clasifica el documento como HUMANO o IA basado en las métricas."""
        votos_humano = 0
        votos_ia = 0
        
        # Perplejidad: Humanos 50-150, IA 20-50
        if 50 <= metricas.get('perplejidad', 0) <= 150:
            votos_humano += 1
        elif 20 <= metricas.get('perplejidad', 0) < 50:
            votos_ia += 1
        
        # Burstiness: Humanos >0.4, IA <0.3
        if metricas.get('burstiness', 0) > 0.4:
            votos_humano += 1
        elif metricas.get('burstiness', 0) < 0.3:
            votos_ia += 1
        
        # TTR: Humanos 0.6-0.8, IA 0.4-0.6
        if 0.6 <= metricas.get('ttr', 0) <= 0.8:
            votos_humano += 1
        elif 0.4 <= metricas.get('ttr', 0) < 0.6:
            votos_ia += 1
        
        # Hapax: Humanos 40-60%, IA 20-40%
        hapax_pct = metricas.get('hapax_ratio', 0) * 100
        if 40 <= hapax_pct <= 60:
            votos_humano += 1
        elif 20 <= hapax_pct < 40:
            votos_ia += 1
        
        # CV MLS: Humanos >0.35
        if metricas.get('cv_mls', 0) > 0.35:
            votos_humano += 1
        elif metricas.get('cv_mls', 0) < 0.25:
            votos_ia += 1
        
        # Coherencia: IA tiende a >0.7 consistente, Humanos 0.4-0.8 variable
        if 0.4 <= metricas.get('coherencia_semantica', 0) <= 0.8:
            if metricas.get('coherencia_varianza', 0) > 0.05:  # Alta variabilidad
                votos_humano += 1
        elif metricas.get('coherencia_semantica', 0) > 0.7 and metricas.get('coherencia_varianza', 0) < 0.03:
            votos_ia += 1
        
        # Ratio función/contenido: Humanos ~0.55, IA ~0.48
        ratio = metricas.get('ratio_funcion_contenido', 0)
        if 0.50 <= ratio <= 0.60:
            votos_humano += 1
        elif 0.45 <= ratio < 0.50:
            votos_ia += 1
        
        return 'HUMANO' if votos_humano > votos_ia else 'IA'
    
    def calcular_probabilidad_humano(self, metricas: Dict) -> float:
        """Calcula probabilidad de que el documento sea humano."""
        score = 0.5  # Base neutral
        
        # Ajustar según métricas
        perp = metricas.get('perplejidad', 50)
        if 50 <= perp <= 150:
            score += 0.15
        elif perp < 20 or perp > 200:
            score -= 0.1
        
        burst = metricas.get('burstiness', 0.3)
        if burst > 0.4:
            score += 0.1
        elif burst < 0.3:
            score -= 0.1
        
        ttr = metricas.get('ttr', 0.5)
        if 0.6 <= ttr <= 0.8:
            score += 0.1
        elif ttr < 0.4:
            score -= 0.1
        
        cv = metricas.get('cv_mls', 0.3)
        if cv > 0.35:
            score += 0.1
        elif cv < 0.25:
            score -= 0.05
        
        return max(0.0, min(1.0, score))

def convertir_a_nativo(obj):
    """Convierte valores numpy a tipos nativos de Python para JSON serialization."""
    import numpy as np
    
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convertir_a_nativo(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convertir_a_nativo(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convertir_a_nativo(item) for item in obj)
    else:
        return obj

def procesar_corpus():
    """Procesa todos los documentos en corpus/documents."""
    detector = DetectorIA()
    carpeta_docs = Path('corpus/documents')
    
    if not carpeta_docs.exists():
        print(f"Error: La carpeta {carpeta_docs} no existe")
        return
    
    # Obtener todos los archivos .txt
    archivos = list(carpeta_docs.glob('*.txt'))
    print(f"Total de documentos encontrados: {len(archivos)}")
    
    resultados = []
    errores = []
    
    # Procesar cada archivo
    for archivo in tqdm(archivos, desc="Analizando documentos"):
        try:
            with open(archivo, 'r', encoding='utf-8', errors='ignore') as f:
                texto = f.read()
            
            resultado = detector.analizar_documento(texto)
            if resultado:
                resultado['documento'] = archivo.stem
                # Convertir valores numpy a tipos nativos
                resultado = convertir_a_nativo(resultado)
                resultados.append(resultado)
        except Exception as e:
            errores.append((archivo.name, str(e)))
    
    # Guardar resultados
    output_dir = Path('corpus/ai_detection_results')
    output_dir.mkdir(exist_ok=True)
    
    # Guardar JSON completo
    with open(output_dir / 'resultados_completos.json', 'w', encoding='utf-8') as f:
        json.dump(resultados, f, indent=2, ensure_ascii=False)
    
    # Generar tabla resumen
    generar_reporte_resumen(resultados, output_dir)
    
    print(f"\nProceso completado:")
    print(f"  Documentos procesados: {len(resultados)}")
    print(f"  Errores: {len(errores)}")
    print(f"  Resultados guardados en: {output_dir}")

def generar_reporte_resumen(resultados: List[Dict], output_dir: Path):
    """Genera reporte resumen en formato CSV y Markdown."""
    if not resultados:
        return
    
    # Preparar datos para tabla
    filas = []
    for r in resultados:
        fila = {
            'Documento': r['documento'],
            'Perplejidad': f"{r['perplejidad']:.2f}",
            'Burstiness': f"{r['burstiness']:.3f}",
            'TTR': f"{r['ttr']:.3f}",
            'MTTR': f"{r['mttr']:.3f}",
            'Hapax_Ratio': f"{r['hapax_ratio']*100:.1f}%",
            'Yule_K': f"{r['yule_k']:.2f}",
            'MLS': f"{r['mls']:.1f}",
            'CV_MLS': f"{r['cv_mls']:.3f}",
            'Flesch_Kincaid': f"{r['flesch_kincaid']:.1f}",
            'FK_Varianza': f"{r['flesch_varianza']:.3f}",
            'Coherencia_Sem': f"{r['coherencia_semantica']:.3f}",
            'Coherencia_Var': f"{r['coherencia_varianza']:.3f}",
            'Ratio_Func_Cont': f"{r['ratio_funcion_contenido']:.3f}",
            'Clasificacion': r['clasificacion'],
            'Probabilidad_Humano': f"{r['probabilidad_humano']*100:.1f}%"
        }
        filas.append(fila)
    
    # Guardar CSV
    import csv
    with open(output_dir / 'tabla_resumen.csv', 'w', encoding='utf-8', newline='') as f:
        if filas:
            writer = csv.DictWriter(f, fieldnames=filas[0].keys(), delimiter=';')
            writer.writeheader()
            writer.writerows(filas)
    
    # Generar estadísticas agregadas
    clasificaciones = Counter([r['clasificacion'] for r in resultados])
    prob_media = statistics.mean([r['probabilidad_humano'] for r in resultados])
    
    # Generar reporte Markdown
    reporte = f"""# Reporte de Detección de Contenido Generado por IA

## Resumen General

- **Total de documentos analizados**: {len(resultados)}
- **Clasificados como HUMANO**: {clasificaciones.get('HUMANO', 0)} ({clasificaciones.get('HUMANO', 0)/len(resultados)*100:.1f}%)
- **Clasificados como IA**: {clasificaciones.get('IA', 0)} ({clasificaciones.get('IA', 0)/len(resultados)*100:.1f}%)
- **Probabilidad promedio de ser humano**: {prob_media*100:.1f}%

## Estadísticas por Métrica

### Perplejidad
- **Media**: {statistics.mean([r['perplejidad'] for r in resultados]):.2f}
- **Rango esperado humano**: 50-150
- **Rango esperado IA**: 20-50

### Burstiness
- **Media**: {statistics.mean([r['burstiness'] for r in resultados]):.3f}
- **Rango esperado humano**: >0.4
- **Rango esperado IA**: <0.3

### Type-Token Ratio (TTR)
- **Media**: {statistics.mean([r['ttr'] for r in resultados]):.3f}
- **Rango esperado humano**: 0.6-0.8
- **Rango esperado IA**: 0.4-0.6

### Hapax Legomena Ratio
- **Media**: {statistics.mean([r['hapax_ratio'] for r in resultados])*100:.1f}%
- **Rango esperado humano**: 40-60%
- **Rango esperado IA**: 20-40%

## Tabla de Resultados

Ver archivo `tabla_resumen.csv` para resultados detallados por documento.

## Notas

- Las métricas están basadas en literatura científica estándar
- La clasificación usa un sistema de votación (voting ensemble)
- La probabilidad de ser humano se calcula como promedio ponderado de métricas
- Algunas métricas pueden requerir modelos adicionales para mayor precisión
"""
    
    with open(output_dir / 'reporte_resumen.md', 'w', encoding='utf-8') as f:
        f.write(reporte)
    
    print(f"\nReportes generados:")
    print(f"  - tabla_resumen.csv")
    print(f"  - reporte_resumen.md")
    print(f"  - resultados_completos.json")

if __name__ == '__main__':
    procesar_corpus()

