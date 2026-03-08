#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Word Embedding Association Test (WEAT) para Análisis de Sesgo de Género en Corpus Médico

Implementa el WEAT de Caliskan et al. (2017) para detectar sesgos de género
en profesiones médicas dentro del corpus sintético.

Referencia principal:
Caliskan, A., Bryson, J. J., & Narayanan, A. (2017). 
Semantics derived automatically from language corpora contain human-like biases. 
Science, 356(6334), 183-186.

Autor: Sistema de Generación de Corpus Sintético
Fecha: Noviembre 2025
"""

import os
import json
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Tuple
from collections import Counter
import re

import numpy as np
from scipy import stats
from tqdm import tqdm

warnings.filterwarnings('ignore')

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class WEATGenderAnalyzer:
    """
    Analizador de sesgo de género usando Word Embedding Association Test (WEAT).
    
    Implementa el test de Caliskan et al. (2017) adaptado para corpus médico español.
    """
    
    def __init__(self, corpus_dir: str):
        """
        Inicializa el analizador WEAT.
        
        Args:
            corpus_dir: Directorio con documentos del corpus
        """
        self.corpus_dir = Path(corpus_dir)
        
        # Listas de palabras para WEAT (español médico)
        # Target sets: Profesiones médicas
        self.medical_professions = [
            'médico', 'medico', 'doctor', 'doctora', 'enfermero', 'enfermera',
            'cirujano', 'cirujana', 'pediatra', 'psiquiatra', 'cardiólogo', 'cardióloga',
            'oncólogo', 'oncóloga', 'neurólogo', 'neuróloga', 'ginecólogo', 'ginecóloga',
            'traumatólogo', 'traumatóloga', 'anestesista', 'radiólogo', 'radióloga',
            'dermatólogo', 'dermatóloga', 'oftalmólogo', 'oftalmóloga',
            'otorrinolaringólogo', 'otorrinolaringóloga', 'urólogo', 'uróloga',
            'endocrinólogo', 'endocrinóloga', 'internista', 'intensivista',
            'residente', 'especialista', 'médica', 'medica'
        ]
        
        # Attribute sets: Género masculino
        self.male_terms = [
            'hombre', 'él', 'masculino', 'varón', 'padre', 'hijo', 'hermano',
            'esposo', 'marido', 'abuelo', 'tío', 'sobrino', 'primo', 'yerno',
            'nieto', 'cuñado', 'padrino', 'suegro'
        ]
        
        # Attribute sets: Género femenino
        self.female_terms = [
            'mujer', 'ella', 'femenino', 'madre', 'hija', 'hermana',
            'esposa', 'abuela', 'tía', 'sobrina', 'prima', 'nuera',
            'nieta', 'cuñada', 'madrina', 'suegra'
        ]
        
        # Cargar documentos
        self.documents = self._load_documents()
        
        # Construir vocabulario
        self.vocab = self._build_vocabulary()
        
        # Entrenar embeddings simples (co-ocurrencia)
        self.embeddings = self._train_simple_embeddings()
    
    def _load_documents(self) -> List[str]:
        """Carga documentos del corpus."""
        logger.info("📂 Cargando documentos para análisis WEAT...")
        documents = []
        txt_files = list(self.corpus_dir.glob("*.txt"))
        
        for txt_file in tqdm(txt_files[:5000], desc="Cargando"):  # Limitar para velocidad
            try:
                with open(txt_file, 'r', encoding='utf-8') as f:
                    text = f.read().strip()
                    if text:
                        documents.append(text)
            except Exception as e:
                logger.warning(f"Error cargando {txt_file}: {e}")
        
        logger.info(f"✅ Cargados {len(documents)} documentos")
        return documents
    
    def _build_vocabulary(self) -> Dict[str, int]:
        """Construye vocabulario del corpus."""
        logger.info("🔤 Construyendo vocabulario...")
        vocab = Counter()
        
        for doc in tqdm(self.documents, desc="Vocabulario"):
            # Tokenización simple
            tokens = re.findall(r'\b\w+\b', doc.lower())
            vocab.update(tokens)
        
        # Filtrar palabras muy raras (< 5 ocurrencias)
        vocab = {word: count for word, count in vocab.items() if count >= 5}
        
        logger.info(f"✅ Vocabulario: {len(vocab)} palabras")
        return vocab
    
    def _train_simple_embeddings(self, window_size: int = 5, dim: int = 100) -> Dict[str, np.ndarray]:
        """
        Entrena embeddings simples basados en co-ocurrencia.
        
        Para un análisis más robusto, se podría usar word2vec pre-entrenado,
        pero esto permite análisis sin dependencias externas.
        
        Args:
            window_size: Tamaño de ventana para co-ocurrencia
            dim: Dimensionalidad de embeddings
            
        Returns:
            Diccionario word -> embedding vector
        """
        logger.info("🧮 Entrenando embeddings de co-ocurrencia...")
        
        # Matriz de co-ocurrencia
        word_to_idx = {word: idx for idx, word in enumerate(self.vocab.keys())}
        idx_to_word = {idx: word for word, idx in word_to_idx.items()}
        
        cooc_matrix = np.zeros((len(word_to_idx), len(word_to_idx)))
        
        # Construir matriz de co-ocurrencia
        for doc in tqdm(self.documents, desc="Co-ocurrencia"):
            tokens = re.findall(r'\b\w+\b', doc.lower())
            tokens = [t for t in tokens if t in word_to_idx]
            
            for i, word in enumerate(tokens):
                start = max(0, i - window_size)
                end = min(len(tokens), i + window_size + 1)
                
                for j in range(start, end):
                    if i != j:
                        word_idx = word_to_idx[word]
                        context_idx = word_to_idx[tokens[j]]
                        cooc_matrix[word_idx, context_idx] += 1
        
        # SVD para reducir dimensionalidad (similar a LSA)
        logger.info("📐 Aplicando SVD para reducción de dimensionalidad...")
        U, S, Vt = np.linalg.svd(cooc_matrix, full_matrices=False)
        
        # Tomar primeros `dim` componentes
        embeddings_matrix = U[:, :dim] * np.sqrt(S[:dim])
        
        # Normalizar embeddings
        norms = np.linalg.norm(embeddings_matrix, axis=1, keepdims=True)
        embeddings_matrix = embeddings_matrix / (norms + 1e-10)
        
        # Crear diccionario
        embeddings = {
            word: embeddings_matrix[idx]
            for word, idx in word_to_idx.items()
        }
        
        logger.info(f"✅ Embeddings entrenados: {len(embeddings)} palabras × {dim} dims")
        return embeddings
    
    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calcula similitud coseno entre dos vectores."""
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-10)
    
    def mean_cosine_similarity(self, word: str, word_list: List[str]) -> float:
        """
        Calcula similitud coseno promedio entre una palabra y una lista de palabras.
        
        Args:
            word: Palabra objetivo
            word_list: Lista de palabras atributo
            
        Returns:
            Similitud coseno promedio
        """
        if word not in self.embeddings:
            return 0.0
        
        similarities = []
        for w in word_list:
            if w in self.embeddings:
                sim = self.cosine_similarity(self.embeddings[word], self.embeddings[w])
                similarities.append(sim)
        
        return np.mean(similarities) if similarities else 0.0
    
    def weat_effect_size(self, target_words: List[str], 
                        attribute1: List[str], attribute2: List[str]) -> float:
        """
        Calcula el efecto de tamaño del WEAT (d de Cohen).
        
        WEAT mide la diferencia en asociación entre:
        - Target words (profesiones médicas)
        - Attribute set 1 (términos masculinos)
        - Attribute set 2 (términos femeninos)
        
        d = mean(s(t, A1, A2)) / std(s(t, A1, A2))
        
        donde s(t, A1, A2) = mean_sim(t, A1) - mean_sim(t, A2)
        
        Args:
            target_words: Palabras objetivo (profesiones)
            attribute1: Atributos grupo 1 (masculino)
            attribute2: Atributos grupo 2 (femenino)
            
        Returns:
            Efecto de tamaño (d de Cohen)
        """
        test_statistics = []
        
        for target in target_words:
            if target in self.embeddings:
                sim_attr1 = self.mean_cosine_similarity(target, attribute1)
                sim_attr2 = self.mean_cosine_similarity(target, attribute2)
                s = sim_attr1 - sim_attr2
                test_statistics.append(s)
        
        if not test_statistics:
            return 0.0
        
        mean_s = np.mean(test_statistics)
        std_s = np.std(test_statistics)
        
        effect_size = mean_s / (std_s + 1e-10)
        return effect_size
    
    def weat_permutation_test(self, target_words: List[str],
                             attribute1: List[str], attribute2: List[str],
                             n_permutations: int = 1000) -> Tuple[float, float]:
        """
        Test de permutación para calcular p-value del WEAT.
        
        Args:
            target_words: Palabras objetivo
            attribute1: Atributos grupo 1
            attribute2: Atributos grupo 2
            n_permutations: Número de permutaciones
            
        Returns:
            (effect_size, p_value)
        """
        # Efecto observado
        observed_effect = self.weat_effect_size(target_words, attribute1, attribute2)
        
        # Permutaciones
        all_attributes = attribute1 + attribute2
        n_attr1 = len(attribute1)
        
        permuted_effects = []
        for _ in range(n_permutations):
            # Permutar atributos
            np.random.shuffle(all_attributes)
            perm_attr1 = all_attributes[:n_attr1]
            perm_attr2 = all_attributes[n_attr1:]
            
            perm_effect = self.weat_effect_size(target_words, perm_attr1, perm_attr2)
            permuted_effects.append(perm_effect)
        
        # Calcular p-value (two-tailed)
        p_value = np.mean(np.abs(permuted_effects) >= np.abs(observed_effect))
        
        return observed_effect, p_value
    
    def analyze_gender_bias(self) -> Dict:
        """
        Ejecuta análisis completo de sesgo de género.
        
        Returns:
            Diccionario con resultados del análisis
        """
        logger.info("\n" + "="*80)
        logger.info("⚖️ ANÁLISIS DE SESGO DE GÉNERO (WEAT)")
        logger.info("="*80)
        
        # WEAT para profesiones médicas
        logger.info("\n📊 Calculando WEAT para profesiones médicas...")
        effect_size, p_value = self.weat_permutation_test(
            target_words=self.medical_professions,
            attribute1=self.male_terms,
            attribute2=self.female_terms,
            n_permutations=1000
        )
        
        logger.info(f"   Effect size (d): {effect_size:.4f}")
        logger.info(f"   p-value: {p_value:.4f}")
        
        # Interpretación
        if abs(effect_size) < 0.2:
            interpretation = "Sesgo negligible"
            emoji = "✅"
        elif abs(effect_size) < 0.5:
            interpretation = "Sesgo pequeño"
            emoji = "⚠️"
        elif abs(effect_size) < 0.8:
            interpretation = "Sesgo moderado"
            emoji = "⚠️⚠️"
        else:
            interpretation = "Sesgo grande"
            emoji = "❌"
        
        if effect_size > 0:
            bias_direction = "masculino"
        else:
            bias_direction = "femenino"
        
        logger.info(f"   {emoji} {interpretation} (dirección: {bias_direction})")
        
        # Análisis adicional: ratio de menciones
        logger.info("\n📈 Analizando ratio de menciones por género...")
        male_count = self._count_mentions(self.male_terms)
        female_count = self._count_mentions(self.female_terms)
        
        total_gender_mentions = male_count + female_count
        male_ratio = male_count / total_gender_mentions if total_gender_mentions > 0 else 0
        female_ratio = female_count / total_gender_mentions if total_gender_mentions > 0 else 0
        
        logger.info(f"   Menciones masculinas: {male_count:,} ({male_ratio*100:.1f}%)")
        logger.info(f"   Menciones femeninas: {female_count:,} ({female_ratio*100:.1f}%)")
        logger.info(f"   Ratio M/F: {male_count/female_count:.2f}" if female_count > 0 else "   Ratio M/F: N/A")
        
        # Análisis de co-ocurrencias con profesiones
        logger.info("\n🔍 Analizando co-ocurrencias profesiones-género...")
        cooc_analysis = self._analyze_profession_gender_cooccurrence()
        
        results = {
            "weat_analysis": {
                "effect_size": float(effect_size),
                "p_value": float(p_value),
                "interpretation": interpretation,
                "bias_direction": bias_direction,
                "significant": bool(p_value < 0.05)
            },
            "mention_ratio": {
                "male_mentions": int(male_count),
                "female_mentions": int(female_count),
                "male_ratio": float(male_ratio),
                "female_ratio": float(female_ratio),
                "male_female_ratio": float(male_count / female_count) if female_count > 0 else None
            },
            "profession_cooccurrence": cooc_analysis
        }
        
        logger.info("\n✅ Análisis de sesgo de género completado")
        return results
    
    def _count_mentions(self, word_list: List[str]) -> int:
        """Cuenta menciones totales de una lista de palabras."""
        total = 0
        for word in word_list:
            if word in self.vocab:
                total += self.vocab[word]
        return total
    
    def _analyze_profession_gender_cooccurrence(self, window: int = 10) -> Dict:
        """
        Analiza co-ocurrencias entre profesiones médicas y términos de género.
        
        Args:
            window: Ventana de palabras para considerar co-ocurrencia
            
        Returns:
            Diccionario con análisis de co-ocurrencias
        """
        male_cooc = 0
        female_cooc = 0
        
        for doc in self.documents:
            tokens = re.findall(r'\b\w+\b', doc.lower())
            
            for i, token in enumerate(tokens):
                if token in self.medical_professions:
                    # Buscar en ventana
                    start = max(0, i - window)
                    end = min(len(tokens), i + window + 1)
                    context = tokens[start:end]
                    
                    # Contar género en contexto
                    for ctx_word in context:
                        if ctx_word in self.male_terms:
                            male_cooc += 1
                        if ctx_word in self.female_terms:
                            female_cooc += 1
        
        total_cooc = male_cooc + female_cooc
        
        return {
            "male_cooccurrences": int(male_cooc),
            "female_cooccurrences": int(female_cooc),
            "male_cooc_ratio": float(male_cooc / total_cooc) if total_cooc > 0 else 0,
            "female_cooc_ratio": float(female_cooc / total_cooc) if total_cooc > 0 else 0
        }


def main():
    """Función principal."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Análisis WEAT de sesgo de género")
    parser.add_argument("--corpus_dir", type=str, default="corpus/documents",
                       help="Directorio con documentos del corpus")
    parser.add_argument("--output", type=str, default="analysis_results/gender_bias_weat.json",
                       help="Archivo de salida para resultados")
    
    args = parser.parse_args()
    
    # Crear analizador
    analyzer = WEATGenderAnalyzer(corpus_dir=args.corpus_dir)
    
    # Ejecutar análisis
    results = analyzer.analyze_gender_bias()
    
    # Guardar resultados
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\n💾 Resultados guardados en: {output_path}")


if __name__ == "__main__":
    main()

