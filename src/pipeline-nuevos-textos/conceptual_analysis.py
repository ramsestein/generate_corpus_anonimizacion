#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ANÁLISIS CONCEPTUAL DEL PIPELINE DE ANONIMIZACIÓN
===================================================
Este script realiza un análisis profundo de patrones de error SIN overfitting léxico.
Se centra en:
- Patrones estructurales (longitud, formato, posición)
- Problemas de segmentación
- Confusiones entre categorías
- Ambigüedades semánticas
- Problemas del contexto en el pipeline

NO utiliza:
- Listas de palabras específicas
- Reglas basadas en tokens concretos
- Heurísticas de vocabulario
"""

import json
import os
import sys
import re
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional

# Añadir al path
sys.path.insert(0, str(Path(__file__).parent))

from entity_fast_filter import EntityFastFilter, EnumDecision


class ConceptualAnalyzer:
    """Analizador de patrones conceptuales sin overfitting léxico."""
    
    def __init__(self, entities_file: str, validation_dir: str):
        """
        Args:
            entities_file: Ruta al JSON con entidades procesadas
            validation_dir: Directorio con resultados de validación LLM
        """
        self.entities_file = entities_file
        self.validation_dir = validation_dir
        self.entities = []
        self.validation_results = {}
        self.filter = None
        
        # Resultados del análisis
        self.analysis_results = {
            "metadata": {},
            "global_metrics": {},
            "metrics_by_label": {},
            "structural_patterns": {},
            "segmentation_issues": [],
            "category_confusions": [],
            "confidence_analysis": {},
            "context_issues": [],
            "recommendations": []
        }
    
    def load_data(self):
        """Carga entidades y resultados de validación."""
        print("\n" + "="*70)
        print("CARGANDO DATOS")
        print("="*70)
        
        # Cargar entidades
        with open(self.entities_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.entities = data.get('entities', [])
        metadata = data.get('metadata', {})
        
        print(f"[INFO] Entidades cargadas: {len(self.entities)}")
        print(f"[INFO] Documentos: {metadata.get('processing_stats', {}).get('documents_processed', 0)}")
        
        # Cargar resultados de validación LLM
        validation_path = Path(self.validation_dir)
        if validation_path.exists():
            for json_file in validation_path.glob("*_verification_result.json"):
                doc_id = json_file.stem.replace("_verification_result", "")
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        self.validation_results[doc_id] = json.load(f)
                except Exception as e:
                    print(f"[WARN] Error cargando {json_file}: {e}")
        
        print(f"[INFO] Resultados de validación cargados: {len(self.validation_results)}")
        
        self.analysis_results["metadata"] = {
            "timestamp": datetime.now().isoformat(),
            "total_entities": len(self.entities),
            "documents": len(set(e['doc_id'] for e in self.entities)),
            "validation_docs": len(self.validation_results)
        }
    
    def init_filter(self):
        """Inicializa el filtro de entidades."""
        print("\n[INFO] Inicializando EntityFastFilter...")
        
        # Rutas a las listas
        base_path = Path(__file__).parent.parent.parent / "data"
        
        self.filter = EntityFastFilter(
            whitelist_paths=[
                str(base_path / "hospitales.json"),
                str(base_path / "lugares.json")
            ],
            blacklist_paths=[
                str(base_path / "medicamentos.json"),
                str(base_path / "patologias.json")
            ],
            cie10_path=str(base_path / "cie10.xls") if (base_path / "cie10.xls").exists() else None
        )
        
        print(f"[INFO] Filtro inicializado")
    
    def analyze_structural_patterns(self):
        """
        Analiza patrones estructurales de las entidades.
        NO mira palabras específicas, solo estructura.
        """
        print("\n" + "="*70)
        print("ANÁLISIS DE PATRONES ESTRUCTURALES")
        print("="*70)
        
        patterns = {
            "by_length": defaultdict(lambda: {"count": 0, "labels": defaultdict(int)}),
            "by_char_type": defaultdict(lambda: {"count": 0, "labels": defaultdict(int)}),
            "by_format": defaultdict(lambda: {"count": 0, "labels": defaultdict(int)}),
            "by_confidence_band": defaultdict(lambda: {"count": 0, "labels": defaultdict(int)}),
            "position_patterns": defaultdict(lambda: {"count": 0, "labels": defaultdict(int)})
        }
        
        for entity in self.entities:
            text = entity['text']
            label = entity['label']
            confidence = entity.get('confidence', 0)
            
            # === PATRÓN 1: Longitud ===
            length = len(text)
            if length == 1:
                length_cat = "1_char"
            elif length <= 3:
                length_cat = "2-3_chars"
            elif length <= 6:
                length_cat = "4-6_chars"
            elif length <= 12:
                length_cat = "7-12_chars"
            else:
                length_cat = "13+_chars"
            
            patterns["by_length"][length_cat]["count"] += 1
            patterns["by_length"][length_cat]["labels"][label] += 1
            
            # === PATRÓN 2: Tipo de caracteres ===
            if text.isdigit():
                char_type = "only_digits"
            elif text.isalpha():
                if text.isupper():
                    char_type = "only_upper"
                elif text.islower():
                    char_type = "only_lower"
                else:
                    char_type = "mixed_case"
            elif re.match(r'^[\W_]+$', text):
                char_type = "only_punctuation"
            elif re.match(r'^[A-Za-z]+\d+$', text) or re.match(r'^\d+[A-Za-z]+$', text):
                char_type = "alphanumeric_code"
            elif re.match(r'^\d+[/\-\.]\d+', text):
                char_type = "numeric_with_separator"
            else:
                char_type = "mixed_content"
            
            patterns["by_char_type"][char_type]["count"] += 1
            patterns["by_char_type"][char_type]["labels"][label] += 1
            
            # === PATRÓN 3: Formato estructural ===
            if re.match(r'^[A-Z]{1,2}\d{2,4}(\.\d+)?$', text):
                format_type = "code_pattern_letter_digits"  # Como códigos CIE
            elif re.match(r'^\d{8,9}[A-Z]?$', text):
                format_type = "dni_nie_pattern"
            elif re.match(r'^\d{9}$', text):
                format_type = "phone_pattern"
            elif re.match(r'^[\w\.-]+@[\w\.-]+\.\w+$', text):
                format_type = "email_pattern"
            elif re.match(r'^https?://', text) or '.es' in text or '.com' in text:
                format_type = "url_pattern"
            elif re.match(r'^\d{1,2}[/\-]\d{1,2}([/\-]\d{2,4})?$', text):
                format_type = "date_pattern"
            elif len(text) <= 3 and text.isupper():
                format_type = "short_abbreviation"
            else:
                format_type = "unstructured"
            
            patterns["by_format"][format_type]["count"] += 1
            patterns["by_format"][format_type]["labels"][label] += 1
            
            # === PATRÓN 4: Banda de confianza ===
            if confidence >= 0.95:
                conf_band = "very_high_95+"
            elif confidence >= 0.85:
                conf_band = "high_85-95"
            elif confidence >= 0.75:
                conf_band = "medium_75-85"
            else:
                conf_band = "low_<75"
            
            patterns["by_confidence_band"][conf_band]["count"] += 1
            patterns["by_confidence_band"][conf_band]["labels"][label] += 1
        
        # Convertir defaultdicts a dicts normales para JSON
        self.analysis_results["structural_patterns"] = {
            k: {k2: {"count": v2["count"], "labels": dict(v2["labels"])} 
                for k2, v2 in v.items()}
            for k, v in patterns.items()
        }
        
        # Imprimir resumen
        print("\n📊 DISTRIBUCIÓN POR LONGITUD:")
        for cat, data in sorted(patterns["by_length"].items()):
            print(f"   {cat}: {data['count']} entidades")
            
        print("\n📊 DISTRIBUCIÓN POR TIPO DE CARACTERES:")
        for cat, data in sorted(patterns["by_char_type"].items()):
            print(f"   {cat}: {data['count']} entidades")
            
        print("\n📊 DISTRIBUCIÓN POR FORMATO:")
        for cat, data in sorted(patterns["by_format"].items()):
            print(f"   {cat}: {data['count']} entidades")
    
    def analyze_segmentation_issues(self):
        """
        Detecta problemas de segmentación del NER.
        Busca patrones como:
        - Entidades fragmentadas (mismo doc, posiciones contiguas)
        - Entidades con límites sospechosos (empiezan/terminan en medio de palabra)
        - Solapamientos entre modelos
        """
        print("\n" + "="*70)
        print("ANÁLISIS DE PROBLEMAS DE SEGMENTACIÓN")
        print("="*70)
        
        issues = []
        
        # Agrupar por documento
        by_doc = defaultdict(list)
        for entity in self.entities:
            by_doc[entity['doc_id']].append(entity)
        
        for doc_id, doc_entities in by_doc.items():
            # Ordenar por posición
            sorted_entities = sorted(doc_entities, key=lambda x: x['start'])
            
            for i, entity in enumerate(sorted_entities):
                # === PROBLEMA 1: Entidad muy corta (posible fragmento) ===
                if len(entity['text']) == 1 and entity['label'] not in ('SEXO_SUJETO_ASISTENCIA',):
                    issues.append({
                        "type": "single_char_fragment",
                        "doc_id": doc_id,
                        "text": entity['text'],
                        "label": entity['label'],
                        "confidence": entity['confidence'],
                        "position": (entity['start'], entity['end']),
                        "model": entity['model'],
                        "description": "Entidad de un solo carácter - probable fragmentación"
                    })
                
                # === PROBLEMA 2: Entidades contiguas del mismo tipo ===
                if i > 0:
                    prev = sorted_entities[i-1]
                    gap = entity['start'] - prev['end']
                    
                    if gap <= 1 and prev['label'] == entity['label']:
                        issues.append({
                            "type": "contiguous_same_label",
                            "doc_id": doc_id,
                            "entities": [
                                {"text": prev['text'], "pos": (prev['start'], prev['end'])},
                                {"text": entity['text'], "pos": (entity['start'], entity['end'])}
                            ],
                            "label": entity['label'],
                            "gap": gap,
                            "description": f"Entidades contiguas ({gap} chars) del mismo tipo - posible entidad fragmentada"
                        })
                    
                    # === PROBLEMA 3: Solapamiento ===
                    if entity['start'] < prev['end']:
                        issues.append({
                            "type": "overlap",
                            "doc_id": doc_id,
                            "entity1": {"text": prev['text'], "label": prev['label'], "model": prev['model']},
                            "entity2": {"text": entity['text'], "label": entity['label'], "model": entity['model']},
                            "overlap_chars": prev['end'] - entity['start'],
                            "description": "Solapamiento entre entidades - conflicto de segmentación"
                        })
                
                # === PROBLEMA 4: Solo puntuación ===
                if re.match(r'^[\W_]+$', entity['text']):
                    issues.append({
                        "type": "punctuation_only",
                        "doc_id": doc_id,
                        "text": entity['text'],
                        "label": entity['label'],
                        "description": "Entidad solo con puntuación - error de tokenización"
                    })
        
        self.analysis_results["segmentation_issues"] = issues
        
        # Contar por tipo
        issue_counts = defaultdict(int)
        for issue in issues:
            issue_counts[issue["type"]] += 1
        
        print(f"\n⚠️ PROBLEMAS DE SEGMENTACIÓN DETECTADOS: {len(issues)}")
        for issue_type, count in sorted(issue_counts.items(), key=lambda x: -x[1]):
            print(f"   {issue_type}: {count}")
    
    def analyze_category_confusions(self):
        """
        Analiza confusiones entre categorías.
        Detecta patrones donde el mismo texto/formato aparece con diferentes etiquetas.
        """
        print("\n" + "="*70)
        print("ANÁLISIS DE CONFUSIONES ENTRE CATEGORÍAS")
        print("="*70)
        
        confusions = []
        
        # Agrupar por formato estructural (no por palabra)
        format_to_labels = defaultdict(lambda: {"labels": set(), "examples": []})
        
        for entity in self.entities:
            text = entity['text']
            
            # Crear "huella estructural" del texto (sin memorizar la palabra)
            fingerprint = self._get_structural_fingerprint(text)
            
            format_to_labels[fingerprint]["labels"].add(entity['label'])
            if len(format_to_labels[fingerprint]["examples"]) < 3:
                format_to_labels[fingerprint]["examples"].append({
                    "text": text,
                    "label": entity['label'],
                    "confidence": entity['confidence']
                })
        
        # Identificar conflictos
        for fingerprint, data in format_to_labels.items():
            if len(data["labels"]) > 1:
                confusions.append({
                    "structural_pattern": fingerprint,
                    "conflicting_labels": list(data["labels"]),
                    "num_labels": len(data["labels"]),
                    "examples": data["examples"],
                    "description": f"Mismo patrón estructural asignado a {len(data['labels'])} categorías diferentes"
                })
        
        # Ordenar por número de conflictos
        confusions.sort(key=lambda x: -x["num_labels"])
        
        self.analysis_results["category_confusions"] = confusions
        
        print(f"\n🔀 PATRONES CON MÚLTIPLES CATEGORÍAS: {len(confusions)}")
        for conf in confusions[:10]:
            print(f"\n   Patrón: {conf['structural_pattern']}")
            print(f"   Etiquetas: {conf['conflicting_labels']}")
            for ex in conf['examples'][:2]:
                print(f"      → '{ex['text']}' ({ex['label']}, conf={ex['confidence']:.3f})")
    
    def _get_structural_fingerprint(self, text: str) -> str:
        """
        Genera una huella estructural del texto sin memorizar palabras específicas.
        Ej: "Dr. García" → "Aa. Aaaaa" (mayúsculas, minúsculas, puntuación)
        """
        result = []
        for char in text:
            if char.isupper():
                result.append('A')
            elif char.islower():
                result.append('a')
            elif char.isdigit():
                result.append('0')
            elif char.isspace():
                result.append(' ')
            else:
                result.append(char)
        
        # Comprimir secuencias repetidas
        compressed = []
        prev = None
        count = 0
        for char in result:
            if char == prev:
                count += 1
            else:
                if prev is not None:
                    if count > 3:
                        compressed.append(f"{prev}+")
                    else:
                        compressed.append(prev * count)
                prev = char
                count = 1
        if prev is not None:
            if count > 3:
                compressed.append(f"{prev}+")
            else:
                compressed.append(prev * count)
        
        return ''.join(compressed)
    
    def analyze_confidence_vs_accuracy(self):
        """
        Analiza la relación entre confianza del NER y tipos de entidades.
        Busca patrones donde alta confianza no implica alta precisión.
        """
        print("\n" + "="*70)
        print("ANÁLISIS DE CONFIANZA VS CATEGORÍA")
        print("="*70)
        
        confidence_analysis = defaultdict(lambda: {
            "total": 0,
            "by_confidence_band": defaultdict(int),
            "avg_confidence": 0,
            "structural_issues": 0
        })
        
        for entity in self.entities:
            label = entity['label']
            conf = entity['confidence']
            text = entity['text']
            
            confidence_analysis[label]["total"] += 1
            confidence_analysis[label]["avg_confidence"] += conf
            
            if conf >= 0.95:
                confidence_analysis[label]["by_confidence_band"]["95+"] += 1
            elif conf >= 0.85:
                confidence_analysis[label]["by_confidence_band"]["85-95"] += 1
            elif conf >= 0.75:
                confidence_analysis[label]["by_confidence_band"]["75-85"] += 1
            else:
                confidence_analysis[label]["by_confidence_band"]["<75"] += 1
            
            # Contar entidades estructuralmente problemáticas
            if len(text) <= 2 or re.match(r'^[\W_]+$', text):
                confidence_analysis[label]["structural_issues"] += 1
        
        # Calcular promedios
        for label, data in confidence_analysis.items():
            if data["total"] > 0:
                data["avg_confidence"] /= data["total"]
                data["structural_issue_rate"] = data["structural_issues"] / data["total"]
        
        self.analysis_results["confidence_analysis"] = {
            k: {
                "total": v["total"],
                "avg_confidence": round(v["avg_confidence"], 4),
                "by_confidence_band": dict(v["by_confidence_band"]),
                "structural_issue_rate": round(v.get("structural_issue_rate", 0), 4)
            }
            for k, v in confidence_analysis.items()
        }
        
        # Ordenar por tasa de problemas estructurales
        sorted_labels = sorted(
            confidence_analysis.items(),
            key=lambda x: -x[1].get("structural_issue_rate", 0)
        )
        
        print("\n📈 ANÁLISIS POR ETIQUETA (ordenado por tasa de problemas):")
        print("-" * 70)
        for label, data in sorted_labels:
            issue_rate = data.get("structural_issue_rate", 0) * 100
            avg_conf = data["avg_confidence"]
            print(f"   {label:40} | Conf.avg={avg_conf:.3f} | Problemas={issue_rate:.1f}%")
    
    def apply_filter_pipeline(self):
        """
        Aplica el filtro a todas las entidades y analiza decisiones.
        """
        print("\n" + "="*70)
        print("APLICANDO PIPELINE DE FILTRADO")
        print("="*70)
        
        filter_results = {
            "by_decision": defaultdict(lambda: {"count": 0, "labels": defaultdict(int)}),
            "by_label_decision": defaultdict(lambda: defaultdict(int)),
            "escalated_to_llm": [],
            "force_ignored": [],
            "force_anonymized": []
        }
        
        for entity in self.entities:
            decision = self.filter.evaluate_candidate(
                entity['text'],
                entity['label']
            )
            
            decision_str = decision.name
            label = entity['label']
            
            filter_results["by_decision"][decision_str]["count"] += 1
            filter_results["by_decision"][decision_str]["labels"][label] += 1
            filter_results["by_label_decision"][label][decision_str] += 1
            
            entity_summary = {
                "text": entity['text'],
                "label": label,
                "confidence": entity['confidence'],
                "doc_id": entity['doc_id'][:8]
            }
            
            if decision == EnumDecision.ESCALATE_TO_LLM:
                filter_results["escalated_to_llm"].append(entity_summary)
            elif decision == EnumDecision.FORCE_IGNORE:
                filter_results["force_ignored"].append(entity_summary)
            elif decision == EnumDecision.FORCE_ANONYMIZE:
                filter_results["force_anonymized"].append(entity_summary)
        
        self.analysis_results["filter_results"] = {
            "by_decision": {
                k: {"count": v["count"], "labels": dict(v["labels"])}
                for k, v in filter_results["by_decision"].items()
            },
            "by_label_decision": {
                k: dict(v) for k, v in filter_results["by_label_decision"].items()
            },
            "escalated_count": len(filter_results["escalated_to_llm"]),
            "ignored_count": len(filter_results["force_ignored"]),
            "anonymized_count": len(filter_results["force_anonymized"])
        }
        
        print("\n🔍 RESULTADOS DEL FILTRADO:")
        print("-" * 70)
        total = len(self.entities)
        for decision, data in sorted(filter_results["by_decision"].items()):
            pct = (data["count"] / total) * 100
            print(f"   {decision:20}: {data['count']:4} ({pct:5.1f}%)")
        
        print("\n📋 DECISIONES POR ETIQUETA:")
        for label, decisions in sorted(filter_results["by_label_decision"].items()):
            total_label = sum(decisions.values())
            print(f"\n   {label}:")
            for dec, count in sorted(decisions.items()):
                pct = (count / total_label) * 100
                print(f"      {dec}: {count} ({pct:.1f}%)")
    
    def generate_conceptual_recommendations(self):
        """
        Genera recomendaciones basadas en patrones conceptuales.
        NO genera reglas léxicas específicas.
        """
        print("\n" + "="*70)
        print("GENERANDO RECOMENDACIONES CONCEPTUALES")
        print("="*70)
        
        recommendations = []
        
        # === RECOMENDACIÓN 1: Problemas de segmentación ===
        segmentation_issues = self.analysis_results.get("segmentation_issues", [])
        single_char_count = sum(1 for i in segmentation_issues if i["type"] == "single_char_fragment")
        
        if single_char_count > 10:
            recommendations.append({
                "category": "SEGMENTACIÓN",
                "priority": "ALTA",
                "issue": f"Alta tasa de entidades de un solo carácter ({single_char_count} casos)",
                "root_cause": "El modelo NER fragmenta identificadores y códigos en tokens individuales",
                "structural_solution": [
                    "Implementar post-procesamiento que fusione entidades contiguas del mismo tipo",
                    "Ajustar el chunking para evitar cortes en medio de identificadores",
                    "Considerar usar modelos NER con tokenización a nivel de palabra completa"
                ],
                "prompt_improvement": [
                    "Instruir al LLM Judge que ignore entidades < 2 caracteres",
                    "Añadir contexto de ventana más amplio (±50 chars) para entidades cortas"
                ]
            })
        
        # === RECOMENDACIÓN 2: Confusiones entre categorías ===
        confusions = self.analysis_results.get("category_confusions", [])
        if len(confusions) > 5:
            # Encontrar las categorías más confundidas
            confused_pairs = defaultdict(int)
            for conf in confusions:
                labels = sorted(conf["conflicting_labels"])
                for i in range(len(labels)):
                    for j in range(i+1, len(labels)):
                        confused_pairs[(labels[i], labels[j])] += 1
            
            top_pairs = sorted(confused_pairs.items(), key=lambda x: -x[1])[:3]
            
            recommendations.append({
                "category": "CONFUSIÓN DE CATEGORÍAS",
                "priority": "ALTA",
                "issue": f"Patrones estructurales asignados a múltiples categorías ({len(confusions)} casos)",
                "top_confusions": [f"{p[0][0]} ↔ {p[0][1]}: {p[1]} casos" for p in top_pairs],
                "root_cause": "Entidades con formato similar reciben etiquetas diferentes según contexto",
                "structural_solution": [
                    "Implementar reglas de desambiguación basadas en contexto sintáctico",
                    "Usar el contexto circundante (preposiciones, verbos) para distinguir",
                    "Crear validadores de formato específicos por etiqueta"
                ],
                "prompt_improvement": [
                    "Incluir en el prompt ejemplos de confusiones frecuentes",
                    "Pedir al LLM que justifique por qué NO es otra categoría",
                    "Añadir contexto de oración completa, no solo fragmento"
                ]
            })
        
        # === RECOMENDACIÓN 3: Entidades con alta confianza pero problemas estructurales ===
        conf_analysis = self.analysis_results.get("confidence_analysis", {})
        problematic_high_conf = []
        for label, data in conf_analysis.items():
            if data.get("avg_confidence", 0) > 0.9 and data.get("structural_issue_rate", 0) > 0.1:
                problematic_high_conf.append(label)
        
        if problematic_high_conf:
            recommendations.append({
                "category": "CALIBRACIÓN DE CONFIANZA",
                "priority": "MEDIA",
                "issue": f"Etiquetas con alta confianza pero alta tasa de problemas estructurales",
                "affected_labels": problematic_high_conf,
                "root_cause": "El NER asigna alta confianza a fragmentos y tokens mal segmentados",
                "structural_solution": [
                    "No usar la confianza del NER como único criterio de filtrado",
                    "Implementar validación de formato mínimo antes de aceptar alta confianza",
                    "Penalizar confianza para entidades muy cortas"
                ],
                "prompt_improvement": [
                    "Instruir al LLM a ser escéptico con entidades cortas aunque tengan alta confianza",
                    "Pedir validación explícita del formato esperado para cada tipo de entidad"
                ]
            })
        
        # === RECOMENDACIÓN 4: Patrones numéricos ambiguos ===
        patterns = self.analysis_results.get("structural_patterns", {})
        numeric_patterns = patterns.get("by_format", {})
        
        code_pattern_labels = numeric_patterns.get("code_pattern_letter_digits", {}).get("labels", {})
        if len(code_pattern_labels) > 1:
            recommendations.append({
                "category": "CÓDIGOS ALFANUMÉRICOS",
                "priority": "MEDIA",
                "issue": "Patrón 'LETRA+DÍGITOS' asignado a múltiples categorías",
                "conflicting_labels": list(code_pattern_labels.keys()),
                "root_cause": "Códigos médicos (CIE-10, ATC) confundidos con identificadores personales",
                "structural_solution": [
                    "Validar formato contra patrones conocidos de códigos médicos",
                    "Verificar si el código existe en catálogos oficiales (CIE-10, etc.)",
                    "Distinguir por posición en el documento (sección diagnóstica vs datos personales)"
                ],
                "prompt_improvement": [
                    "Instruir al LLM sobre diferencias entre códigos médicos e identificadores",
                    "Proporcionar contexto de sección del documento",
                    "Pedir que verifique si parece un código de diagnóstico"
                ]
            })
        
        # === RECOMENDACIÓN 5: Contexto insuficiente ===
        filter_results = self.analysis_results.get("filter_results", {})
        escalated_pct = filter_results.get("escalated_count", 0) / len(self.entities) * 100 if self.entities else 0
        
        if escalated_pct > 30:
            recommendations.append({
                "category": "CONTEXTO",
                "priority": "ALTA",
                "issue": f"Alta tasa de escalado a LLM ({escalated_pct:.1f}%)",
                "root_cause": "El filtro determinista no tiene suficiente información para decidir",
                "structural_solution": [
                    "Ampliar las listas de términos médicos conocidos (sin memorizar palabras del test)",
                    "Implementar detección de patrones de sección (Diagnóstico, Datos personales, etc.)",
                    "Usar contexto de ventana más amplio en el filtro"
                ],
                "prompt_improvement": [
                    "Proporcionar al LLM el contexto de oración completa",
                    "Incluir información sobre la sección del documento",
                    "Pedir razonamiento explícito antes de la decisión"
                ]
            })
        
        self.analysis_results["recommendations"] = recommendations
        
        print(f"\n📝 RECOMENDACIONES GENERADAS: {len(recommendations)}")
        for i, rec in enumerate(recommendations, 1):
            print(f"\n{'='*70}")
            print(f"[{rec['priority']}] {i}. {rec['category']}")
            print(f"{'='*70}")
            print(f"Problema: {rec['issue']}")
            print(f"Causa raíz: {rec['root_cause']}")
            print("\nSoluciones estructurales:")
            for sol in rec.get('structural_solution', []):
                print(f"   • {sol}")
            print("\nMejoras en prompts:")
            for sol in rec.get('prompt_improvement', []):
                print(f"   • {sol}")
    
    def generate_report(self, output_dir: str):
        """Genera el informe final en Markdown y JSON."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # === JSON ===
        json_path = output_path / "conceptual_analysis.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.analysis_results, f, indent=2, ensure_ascii=False, default=str)
        
        # === MARKDOWN ===
        md_path = output_path / "conceptual_analysis.md"
        
        md_content = f"""# Análisis Conceptual del Pipeline de Anonimización

**Fecha:** {datetime.now().strftime('%Y-%m-%d %H:%M')}

---

## 1. Resumen Ejecutivo

Este análisis identifica patrones de error **sin crear reglas basadas en palabras específicas**.
Se centra exclusivamente en:

- Problemas estructurales (longitud, formato, segmentación)
- Confusiones entre categorías
- Calibración de confianza
- Insuficiencia de contexto

### 1.1 Dataset Analizado

| Métrica | Valor |
|---------|-------|
| Total entidades | {self.analysis_results['metadata']['total_entities']} |
| Documentos | {self.analysis_results['metadata']['documents']} |
| Resultados validación | {self.analysis_results['metadata']['validation_docs']} |

---

## 2. Patrones Estructurales

### 2.1 Distribución por Longitud

"""
        patterns = self.analysis_results.get("structural_patterns", {})
        by_length = patterns.get("by_length", {})
        for cat, data in sorted(by_length.items()):
            md_content += f"- **{cat}**: {data['count']} entidades\n"
            for label, count in sorted(data['labels'].items(), key=lambda x: -x[1])[:3]:
                md_content += f"  - {label}: {count}\n"
        
        md_content += """
### 2.2 Distribución por Tipo de Caracteres

"""
        by_char = patterns.get("by_char_type", {})
        for cat, data in sorted(by_char.items()):
            md_content += f"- **{cat}**: {data['count']} entidades\n"
        
        md_content += """
---

## 3. Problemas de Segmentación

"""
        segmentation = self.analysis_results.get("segmentation_issues", [])
        issue_counts = defaultdict(int)
        for issue in segmentation:
            issue_counts[issue["type"]] += 1
        
        md_content += f"**Total problemas detectados:** {len(segmentation)}\n\n"
        for issue_type, count in sorted(issue_counts.items(), key=lambda x: -x[1]):
            md_content += f"- **{issue_type}**: {count} casos\n"
        
        md_content += """
---

## 4. Confusiones entre Categorías

"""
        confusions = self.analysis_results.get("category_confusions", [])
        md_content += f"**Patrones con múltiples categorías:** {len(confusions)}\n\n"
        
        for conf in confusions[:5]:
            md_content += f"### Patrón: `{conf['structural_pattern']}`\n"
            md_content += f"Etiquetas: {', '.join(conf['conflicting_labels'])}\n\n"
        
        md_content += """
---

## 5. Análisis de Confianza

"""
        conf_analysis = self.analysis_results.get("confidence_analysis", {})
        md_content += "| Etiqueta | Conf. Promedio | Tasa Problemas |\n"
        md_content += "|----------|----------------|----------------|\n"
        for label, data in sorted(conf_analysis.items(), key=lambda x: -x[1].get("structural_issue_rate", 0)):
            md_content += f"| {label} | {data['avg_confidence']:.3f} | {data['structural_issue_rate']*100:.1f}% |\n"
        
        md_content += """
---

## 6. Resultados del Filtrado

"""
        filter_res = self.analysis_results.get("filter_results", {})
        by_dec = filter_res.get("by_decision", {})
        total = sum(d["count"] for d in by_dec.values())
        
        md_content += "| Decisión | Cantidad | Porcentaje |\n"
        md_content += "|----------|----------|------------|\n"
        for dec, data in sorted(by_dec.items()):
            pct = (data["count"] / total * 100) if total > 0 else 0
            md_content += f"| {dec} | {data['count']} | {pct:.1f}% |\n"
        
        md_content += """
---

## 7. Recomendaciones (Sin Overfitting Léxico)

"""
        recommendations = self.analysis_results.get("recommendations", [])
        for i, rec in enumerate(recommendations, 1):
            md_content += f"""
### 7.{i} [{rec['priority']}] {rec['category']}

**Problema:** {rec['issue']}

**Causa raíz:** {rec['root_cause']}

**Soluciones estructurales:**
"""
            for sol in rec.get('structural_solution', []):
                md_content += f"- {sol}\n"
            
            md_content += "\n**Mejoras en prompts:**\n"
            for sol in rec.get('prompt_improvement', []):
                md_content += f"- {sol}\n"
        
        md_content += """
---

## 8. Conclusiones

Este análisis ha identificado patrones de error basándose únicamente en:

1. **Características estructurales** (longitud, formato, tipo de caracteres)
2. **Problemas de segmentación** (fragmentación, solapamientos)
3. **Confusiones de categoría** (mismos patrones con etiquetas diferentes)
4. **Calibración de confianza** (alta confianza en entidades problemáticas)

**NO se han generado reglas basadas en:**
- Palabras específicas del dataset
- Vocabulario memorizado
- Tokens concretos

Las recomendaciones se centran en mejoras estructurales del pipeline, prompts, y estrategias de razonamiento.

---

*Informe generado automáticamente - Análisis conceptual sin overfitting léxico*
"""
        
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(md_content)
        
        print(f"\n[INFO] Informe guardado en: {md_path}")
        print(f"[INFO] Datos JSON guardados en: {json_path}")
        
        return md_path, json_path
    
    def run_full_analysis(self):
        """Ejecuta el análisis completo."""
        print("\n" + "="*70)
        print("ANÁLISIS CONCEPTUAL DEL PIPELINE")
        print("(Sin overfitting léxico)")
        print("="*70)
        
        # 1. Cargar datos
        self.load_data()
        
        # 2. Inicializar filtro
        self.init_filter()
        
        # 3. Análisis estructural
        self.analyze_structural_patterns()
        
        # 4. Problemas de segmentación
        self.analyze_segmentation_issues()
        
        # 5. Confusiones entre categorías
        self.analyze_category_confusions()
        
        # 6. Análisis de confianza
        self.analyze_confidence_vs_accuracy()
        
        # 7. Aplicar filtro
        self.apply_filter_pipeline()
        
        # 8. Generar recomendaciones
        self.generate_conceptual_recommendations()
        
        # 9. Generar informe
        output_dir = Path(__file__).parent.parent.parent / "reports"
        self.generate_report(str(output_dir))
        
        return self.analysis_results


def main():
    """Punto de entrada principal."""
    base_path = Path(__file__).parent.parent.parent
    
    entities_file = base_path / "entidades-procesadas-para-metricas.json"
    validation_dir = base_path / "step6_validation_judgeLLM"
    
    if not entities_file.exists():
        print(f"[ERROR] No se encuentra el archivo: {entities_file}")
        return
    
    analyzer = ConceptualAnalyzer(
        entities_file=str(entities_file),
        validation_dir=str(validation_dir)
    )
    
    results = analyzer.run_full_analysis()
    
    print("\n" + "="*70)
    print("ANÁLISIS COMPLETADO")
    print("="*70)


if __name__ == "__main__":
    main()
