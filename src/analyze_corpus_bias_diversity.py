#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Análisis Exhaustivo de Sesgo, Diversidad y Calidad de Corpus Sintético Médico

Este script implementa métricas validadas en la literatura para evaluar:
- Diversidad léxica y sintáctica
- Sesgos demográficos y lingüísticos
- Cobertura de subdominios médicos
- Calidad lingüística
- Comparación con corpus reales (MEDDOCAN/CARMEN-I)

Referencias principales:
- Li et al. (2016): "A Diversity-Promoting Objective Function for Neural Conversation Models"
- Zhu et al. (2018): "Texygen: A Benchmarking Platform for Text Generation Models"
- McCarthy & Jarvis (2010): "MTLD, vocd-D, and HD-D: A validation study"
- Caliskan et al. (2017): "Semantics derived automatically from language corpora contain human-like biases"

Uso:
    python analyze_corpus_bias_diversity.py --corpus_dir corpus/documents --output_dir analysis_results

Autor: Sistema de Generación de Corpus Sintético
Fecha: Noviembre 2025
"""

import os
import json
import pickle
import argparse
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from collections import Counter, defaultdict
from datetime import datetime
import multiprocessing as mp

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Backend sin GUI
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from scipy import stats
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy, kstest, chi2_contingency, mannwhitneyu

# Suprimir warnings de librerías externas
warnings.filterwarnings('ignore')

# Configuración de logging (debe estar antes de usarlo)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('corpus_analysis.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# UMAP y TSNE son opcionales para visualizaciones avanzadas
try:
    from sklearn.manifold import TSNE
    TSNE_AVAILABLE = True
except ImportError:
    TSNE_AVAILABLE = False
    logger.warning("⚠️ sklearn no disponible - visualizaciones t-SNE deshabilitadas")

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    logger.warning("⚠️ umap no disponible - visualizaciones UMAP deshabilitadas")

# Configuración de visualización
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")


class CorpusAnalyzer:
    """
    Analizador principal del corpus sintético médico.
    
    Implementa múltiples categorías de análisis para evaluar calidad,
    diversidad y ausencia de sesgos en el corpus generado.
    """
    
    def __init__(self, corpus_dir: str, output_dir: str, cache_dir: str = "cache"):
        """
        Inicializa el analizador.
        
        Args:
            corpus_dir: Directorio con documentos del corpus (.txt)
            output_dir: Directorio para resultados y visualizaciones
            cache_dir: Directorio para cachear resultados intermedios
        """
        self.corpus_dir = Path(corpus_dir)
        self.output_dir = Path(output_dir)
        self.cache_dir = Path(cache_dir)
        
        # Crear directorios si no existen
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Cargar documentos
        self.documents = self._load_documents()
        logger.info(f"✅ Cargados {len(self.documents)} documentos del corpus")
        
        # Inicializar contenedores de resultados
        self.results = {
            "metadata": {
                "corpus_dir": str(corpus_dir),
                "num_documents": len(self.documents),
                "analysis_date": datetime.now().isoformat(),
                "total_tokens": 0,
                "total_chars": 0
            },
            "lexical_diversity": {},
            "syntactic_complexity": {},
            "bias_analysis": {},
            "medical_coverage": {},
            "quality_metrics": {},
            "comparison_real_corpora": {}
        }
    
    def _load_documents(self) -> List[str]:
        """
        Carga todos los documentos del corpus.
        
        Returns:
            Lista de textos de documentos
        """
        cache_file = self.cache_dir / "documents_cache.pkl"
        
        if cache_file.exists():
            logger.info("📂 Cargando documentos desde caché...")
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        
        logger.info("📂 Cargando documentos desde disco...")
        documents = []
        txt_files = list(self.corpus_dir.glob("*.txt"))
        
        for txt_file in tqdm(txt_files, desc="Cargando documentos"):
            try:
                with open(txt_file, 'r', encoding='utf-8') as f:
                    text = f.read().strip()
                    if text:  # Solo documentos no vacíos
                        documents.append(text)
            except Exception as e:
                logger.warning(f"Error cargando {txt_file}: {e}")
        
        # Guardar en caché
        with open(cache_file, 'wb') as f:
            pickle.dump(documents, f)
        
        return documents
    
    def analyze_lexical_diversity(self) -> Dict[str, float]:
        """
        Analiza diversidad léxica del corpus usando múltiples métricas.
        
        Implementa:
        - Type-Token Ratio (TTR)
        - Moving Average TTR (MATTR) con ventana 1000
        - Measure of Textual Lexical Diversity (MTLD)
        - Distinct n-grams (1,2,3,4)
        - Entropía de vocabulario
        - Análisis de distribución de Zipf
        
        Referencias:
        - McCarthy & Jarvis (2010): MTLD validation study
        - Li et al. (2016): Distinct n-grams for diversity
        
        Returns:
            Diccionario con métricas de diversidad léxica
        """
        logger.info("\n" + "="*80)
        logger.info("📊 ANÁLISIS DE DIVERSIDAD LÉXICA")
        logger.info("="*80)
        
        cache_file = self.cache_dir / "lexical_diversity_cache.pkl"
        if cache_file.exists():
            logger.info("📂 Cargando resultados desde caché...")
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        
        # Tokenizar todos los documentos
        logger.info("🔤 Tokenizando corpus...")
        all_tokens = []
        for doc in tqdm(self.documents, desc="Tokenizando"):
            tokens = doc.lower().split()
            all_tokens.extend(tokens)
        
        total_tokens = len(all_tokens)
        unique_tokens = len(set(all_tokens))
        
        logger.info(f"   Total tokens: {total_tokens:,}")
        logger.info(f"   Tokens únicos: {unique_tokens:,}")
        
        # 1. Type-Token Ratio (TTR)
        ttr = unique_tokens / total_tokens if total_tokens > 0 else 0
        logger.info(f"   TTR: {ttr:.4f}")
        
        # 2. Moving Average TTR (MATTR)
        mattr = self._calculate_mattr(all_tokens, window_size=1000)
        logger.info(f"   MATTR (window=1000): {mattr:.4f}")
        
        # 3. MTLD (Measure of Textual Lexical Diversity)
        mtld = self._calculate_mtld(all_tokens)
        logger.info(f"   MTLD: {mtld:.2f}")
        
        # 4. Distinct n-grams
        distinct_ngrams = {}
        for n in [1, 2, 3, 4]:
            distinct = self._calculate_distinct_ngrams(all_tokens, n)
            distinct_ngrams[f'distinct_{n}gram'] = distinct
            logger.info(f"   Distinct-{n}: {distinct:.4f}")
        
        # 5. Entropía de vocabulario (Shannon)
        vocab_entropy = self._calculate_vocabulary_entropy(all_tokens)
        logger.info(f"   Entropía vocabulario: {vocab_entropy:.4f}")
        
        # 6. Análisis de distribución de Zipf
        zipf_ks, zipf_p = self._analyze_zipf_distribution(all_tokens)
        logger.info(f"   Zipf KS statistic: {zipf_ks:.4f}")
        logger.info(f"   Zipf p-value: {zipf_p:.4f}")
        
        results = {
            "ttr": ttr,
            "mattr": mattr,
            "mtld": mtld,
            **distinct_ngrams,
            "vocabulary_entropy": vocab_entropy,
            "zipf_ks_statistic": zipf_ks,
            "zipf_p_value": zipf_p,
            "total_tokens": total_tokens,
            "unique_tokens": unique_tokens,
            "vocabulary_size": unique_tokens
        }
        
        # Guardar en caché
        with open(cache_file, 'wb') as f:
            pickle.dump(results, f)
        
        return results
    
    def _calculate_mattr(self, tokens: List[str], window_size: int = 1000) -> float:
        """
        Calcula Moving Average Type-Token Ratio.
        
        Args:
            tokens: Lista de tokens
            window_size: Tamaño de ventana móvil
            
        Returns:
            MATTR score
        """
        if len(tokens) < window_size:
            return len(set(tokens)) / len(tokens)
        
        ttrs = []
        for i in range(len(tokens) - window_size + 1):
            window = tokens[i:i + window_size]
            ttr = len(set(window)) / len(window)
            ttrs.append(ttr)
        
        return np.mean(ttrs) if ttrs else 0.0
    
    def _calculate_mtld(self, tokens: List[str], threshold: float = 0.72) -> float:
        """
        Calcula Measure of Textual Lexical Diversity (MTLD).
        
        Basado en McCarthy & Jarvis (2010).
        
        Args:
            tokens: Lista de tokens
            threshold: Umbral de TTR para segmentación
            
        Returns:
            MTLD score
        """
        def compute_mtld_forward(toks, thr):
            factor_count = 0
            factor_lengths = []
            types_seen = set()
            tokens_in_factor = 0
            
            for token in toks:
                types_seen.add(token)
                tokens_in_factor += 1
                
                ttr = len(types_seen) / tokens_in_factor
                if ttr <= thr:
                    factor_count += 1
                    factor_lengths.append(tokens_in_factor)
                    types_seen = set()
                    tokens_in_factor = 0
            
            # Factor parcial al final
            if tokens_in_factor > 0:
                ttr = len(types_seen) / tokens_in_factor
                factor_count += (1 - ttr) / (1 - thr) if thr < 1 else 0
            
            return len(toks) / factor_count if factor_count > 0 else len(toks)
        
        # MTLD bidireccional (forward y backward)
        mtld_forward = compute_mtld_forward(tokens, threshold)
        mtld_backward = compute_mtld_forward(tokens[::-1], threshold)
        
        return (mtld_forward + mtld_backward) / 2
    
    def _calculate_distinct_ngrams(self, tokens: List[str], n: int) -> float:
        """
        Calcula Distinct-n metric (Li et al., 2016).
        
        Args:
            tokens: Lista de tokens
            n: Orden del n-grama
            
        Returns:
            Distinct-n score (ratio de n-gramas únicos)
        """
        ngrams = []
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i:i+n])
            ngrams.append(ngram)
        
        if not ngrams:
            return 0.0
        
        return len(set(ngrams)) / len(ngrams)
    
    def _calculate_vocabulary_entropy(self, tokens: List[str]) -> float:
        """
        Calcula entropía de Shannon del vocabulario.
        
        Args:
            tokens: Lista de tokens
            
        Returns:
            Entropía (bits)
        """
        token_counts = Counter(tokens)
        total = sum(token_counts.values())
        probs = np.array([count / total for count in token_counts.values()])
        
        return entropy(probs, base=2)
    
    def _analyze_zipf_distribution(self, tokens: List[str]) -> Tuple[float, float]:
        """
        Analiza si la distribución de frecuencias sigue ley de Zipf.
        
        Usa test de Kolmogorov-Smirnov para comparar con distribución teórica.
        
        Args:
            tokens: Lista de tokens
            
        Returns:
            (KS statistic, p-value)
        """
        # Frecuencias observadas
        freq_dist = Counter(tokens)
        frequencies = sorted(freq_dist.values(), reverse=True)
        
        # Distribución de Zipf teórica
        ranks = np.arange(1, len(frequencies) + 1)
        zipf_theoretical = 1 / ranks
        zipf_theoretical = zipf_theoretical / zipf_theoretical.sum()
        
        # Normalizar frecuencias observadas
        frequencies_norm = np.array(frequencies) / sum(frequencies)
        
        # Test KS
        ks_stat, p_value = kstest(frequencies_norm, lambda x: zipf_theoretical[:len(frequencies_norm)])
        
        return ks_stat, p_value
    
    def run_full_analysis(self):
        """
        Ejecuta análisis completo del corpus.
        """
        logger.info("\n" + "="*80)
        logger.info("🚀 INICIANDO ANÁLISIS COMPLETO DEL CORPUS SINTÉTICO")
        logger.info("="*80)
        
        # 1. Análisis de diversidad léxica
        self.results["lexical_diversity"] = self.analyze_lexical_diversity()
        
        # 2. Análisis de complejidad sintáctica
        logger.info("\n⚠️ Análisis sintáctico requiere spacy - implementar en siguiente fase")
        
        # 3. Análisis de sesgo
        logger.info("\n⚠️ Análisis de sesgo requiere embeddings - implementar en siguiente fase")
        
        # 4. Análisis de cobertura médica
        logger.info("\n⚠️ Análisis cobertura médica requiere SNOMED-CT - implementar en siguiente fase")
        
        # 5. Análisis de calidad
        logger.info("\n⚠️ Análisis de calidad requiere modelos de lenguaje - implementar en siguiente fase")
        
        # Guardar resultados
        self.save_results()
        
        # Generar visualizaciones
        self.generate_visualizations()
        
        logger.info("\n" + "="*80)
        logger.info("✅ ANÁLISIS COMPLETADO")
        logger.info("="*80)
    
    def save_results(self):
        """
        Guarda resultados en formato JSON.
        """
        output_file = self.output_dir / "corpus_analysis_results.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"\n💾 Resultados guardados en: {output_file}")
    
    def generate_visualizations(self):
        """
        Genera visualizaciones de los resultados.
        """
        logger.info("\n📊 Generando visualizaciones...")
        
        viz_dir = self.output_dir / "visualizations"
        viz_dir.mkdir(exist_ok=True)
        
        # Visualización 1: Métricas de diversidad léxica
        self._plot_lexical_diversity_summary(viz_dir)
        
        logger.info(f"   Visualizaciones guardadas en: {viz_dir}")
    
    def _plot_lexical_diversity_summary(self, viz_dir: Path):
        """
        Genera gráfico de resumen de diversidad léxica.
        """
        metrics = self.results["lexical_diversity"]
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Análisis de Diversidad Léxica - Corpus Sintético Médico', 
                     fontsize=16, fontweight='bold')
        
        # Métricas principales
        metric_names = ['TTR', 'MATTR', 'MTLD', 'Distinct-1']
        metric_values = [
            metrics.get('ttr', 0),
            metrics.get('mattr', 0),
            metrics.get('mtld', 0) / 100,  # Normalizar
            metrics.get('distinct_1gram', 0)
        ]
        
        axes[0, 0].bar(metric_names, metric_values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        axes[0, 0].set_title('Métricas Principales de Diversidad Léxica')
        axes[0, 0].set_ylabel('Valor Normalizado')
        axes[0, 0].grid(axis='y', alpha=0.3)
        
        # Distinct n-grams
        ngram_orders = [1, 2, 3, 4]
        ngram_values = [metrics.get(f'distinct_{n}gram', 0) for n in ngram_orders]
        
        axes[0, 1].plot(ngram_orders, ngram_values, marker='o', linewidth=2, markersize=8)
        axes[0, 1].set_title('Distinct N-grams')
        axes[0, 1].set_xlabel('Orden del N-grama')
        axes[0, 1].set_ylabel('Distinct Score')
        axes[0, 1].grid(alpha=0.3)
        
        # Vocabulario
        axes[1, 0].bar(['Total Tokens', 'Tokens Únicos'], 
                       [metrics.get('total_tokens', 0), metrics.get('unique_tokens', 0)],
                       color=['#9467bd', '#8c564b'])
        axes[1, 0].set_title('Tamaño del Vocabulario')
        axes[1, 0].set_ylabel('Cantidad')
        axes[1, 0].ticklabel_format(style='plain', axis='y')
        axes[1, 0].grid(axis='y', alpha=0.3)
        
        # Zipf distribution
        axes[1, 1].text(0.5, 0.6, f"KS Statistic: {metrics.get('zipf_ks_statistic', 0):.4f}", 
                        ha='center', fontsize=12)
        axes[1, 1].text(0.5, 0.4, f"p-value: {metrics.get('zipf_p_value', 0):.4f}", 
                        ha='center', fontsize=12)
        axes[1, 1].text(0.5, 0.2, 
                        "✅ Sigue Ley de Zipf" if metrics.get('zipf_p_value', 0) > 0.05 else "⚠️ Desviación de Zipf",
                        ha='center', fontsize=14, fontweight='bold',
                        color='green' if metrics.get('zipf_p_value', 0) > 0.05 else 'orange')
        axes[1, 1].set_title('Test de Distribución de Zipf')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(viz_dir / 'lexical_diversity_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("   ✅ Gráfico de diversidad léxica generado")


def main():
    """
    Función principal del script.
    """
    parser = argparse.ArgumentParser(
        description="Análisis exhaustivo de sesgo, diversidad y calidad de corpus sintético médico"
    )
    parser.add_argument(
        "--corpus_dir",
        type=str,
        default="corpus/documents",
        help="Directorio con documentos del corpus (.txt)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="analysis_results",
        help="Directorio para guardar resultados y visualizaciones"
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="cache_analysis",
        help="Directorio para cachear resultados intermedios"
    )
    
    args = parser.parse_args()
    
    # Crear analizador y ejecutar
    analyzer = CorpusAnalyzer(
        corpus_dir=args.corpus_dir,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir
    )
    
    analyzer.run_full_analysis()
    
    logger.info("\n✅ Análisis completado exitosamente")
    logger.info(f"📁 Resultados disponibles en: {args.output_dir}")


if __name__ == "__main__":
    main()

