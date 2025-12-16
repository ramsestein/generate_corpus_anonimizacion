#!/usr/bin/env python3
"""
SetFit Gatekeeper - Clasificador principal PII vs Ruido (SIMPLIFICADO)
=====================================================================

VERSIÓN SIMPLIFICADA:
- SetFit clasifica TODO por igual (sin filtros de ruido predefinidos)
- Solo mantiene patrones regex para PII obvio (opcional)
- Umbral de confianza más alto para mejorar precisión

El modelo SetFit debe aprender a distinguir PII de ruido por sí mismo.
"""

import re
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# ============================================================================
# PATRONES REGEX PARA PII OBVIO (opcional, solo si enable_pii_detector=True)
# ============================================================================

COMPILED_PII_PATTERNS = {
    'email': re.compile(r'^[\w.+-]+@[\w-]+\.[\w.-]+$'),
    'phone_es': re.compile(r'^(\+34\s?)?(6|7|8|9)\d{2}[\s.-]?\d{3}[\s.-]?\d{3}$'),
    'phone_intl': re.compile(r'^\+\d{1,3}[\s.-]?\d{3,4}[\s.-]?\d{3,4}[\s.-]?\d{3,4}$'),
    'dni_nif': re.compile(r'^[0-9]{8}[A-Z]$'),
    'nie': re.compile(r'^[XYZ][0-9]{7}[A-Z]$'),
    'iban_es': re.compile(r'^ES\d{22}$'),
    'credit_card': re.compile(r'^\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}$'),
    'ss_number': re.compile(r'^\d{12}$'),
    'postal_code': re.compile(r'^\d{5}$'),
    'license_plate': re.compile(r'^[0-9]{4}[A-Z]{3}$|^[A-Z]{1,2}[0-9]{4}[A-Z]{2}$'),
    'full_date': re.compile(r'^\d{1,2}[/-]\d{1,2}[/-]\d{2,4}$'),
}

# ============================================================================
# SINGLETON GLOBAL PARA EL MODELO SETFIT
# ============================================================================

_MODEL_INSTANCE = None
_MODEL_PATH_CACHED = None


def get_model_instance(model_path: str):
    """Carga el modelo SetFit una sola vez (singleton)."""
    global _MODEL_INSTANCE, _MODEL_PATH_CACHED
    
    if _MODEL_INSTANCE is not None and _MODEL_PATH_CACHED == model_path:
        return _MODEL_INSTANCE
    
    try:
        from setfit import SetFitModel
    except ImportError:
        raise ImportError("SetFit no está instalado. Ejecuta: pip install setfit")
    
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Modelo no encontrado: {model_path}")
    
    logger.debug(f"Cargando modelo SetFit desde {model_path}")
    _MODEL_INSTANCE = SetFitModel.from_pretrained(str(model_path))
    _MODEL_PATH_CACHED = model_path
    
    return _MODEL_INSTANCE


# ============================================================================
# CLASES Y FUNCIONES
# ============================================================================

@dataclass
class ClassificationResult:
    """Resultado de la clasificación de una entidad."""
    is_pii: bool
    confidence: float
    classification_method: str
    original_label: str
    entity_text: str
    details: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte a diccionario."""
        return {
            "is_pii": self.is_pii,
            "confidence": self.confidence,
            "classification_method": self.classification_method,
            "original_label": self.original_label,
            "entity_text": self.entity_text,
            "details": self.details,
        }


class SetFitGatekeeper:
    """
    Clasificador SetFit SIMPLIFICADO.
    
    Cambios principales respecto a la versión anterior:
    - NO hay filtros de ruido predefinidos (COMMON_WORDS eliminado)
    - NO hay always_pii_labels (todo pasa por SetFit igual)
    - SetFit clasifica TODO por igual
    - Umbral de confianza más alto por defecto (0.85) para mejor precisión
    """
    
    def __init__(
        self,
        model_path: str = "models/gatekeeper_setfit",
        confidence_threshold: float = 0.85,  # Subido de 0.75 a 0.85
        enable_pii_detector: bool = False,   # Detector de PII obvio (regex) - opcional
        enable_low_confidence_filter: bool = True,  # Filtrar baja confianza
    ):
        """Inicializa el gatekeeper con el modelo SetFit."""
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.enable_pii_detector = enable_pii_detector
        self.enable_low_confidence_filter = enable_low_confidence_filter
        
        # Modelo cargado de forma diferida (lazy loading)
        self._model = None
        self._model_loaded = False
        
        # Estadísticas
        self.stats = {
            'total_processed': 0,
            'pii_detected_regex': 0,
            'low_confidence_filtered': 0,
            'setfit_pii': 0,
            'setfit_noise': 0,
        }
        
        logger.debug(f"SetFitGatekeeper configurado con umbral={confidence_threshold}")
    
    @property
    def model(self):
        """Carga diferida del modelo SetFit (singleton)."""
        if not self._model_loaded:
            self._model = get_model_instance(self.model_path)
            self._model_loaded = True
        return self._model
    
    def _is_obvious_pii(self, entity_text: str, entity_label: str) -> Tuple[bool, str]:
        """Detecta PII obvio usando patrones regex (opcional)."""
        for pattern_name, pattern in COMPILED_PII_PATTERNS.items():
            if pattern.match(entity_text.strip()):
                return True, pattern_name
        
        # Detector de nombres compuestos (caso especial)
        if entity_label in ['NOMBRE_SUJETO_ASISTENCIA', 'NOMBRE_PERSONAL_SANITARIO']:
            words = entity_text.split()
            if len(words) >= 2 and all(w[0].isupper() for w in words if len(w) > 1):
                return True, "compound_name"
        
        return False, ""
    
    def _classify_with_setfit(self, input_text: str) -> Tuple[int, float]:
        """Clasifica usando SetFit."""
        prediction = self.model.predict([input_text])[0]
        
        try:
            probabilities = self.model.predict_proba([input_text])[0]
            confidence = float(max(probabilities))
        except Exception:
            confidence = 1.0
        
        return int(prediction), confidence
    
    def classify(
        self,
        entity_text: str,
        entity_label: str,
        sentence_context: str,
        full_document: Optional[str] = None,
        other_entities: Optional[List[str]] = None,
    ) -> ClassificationResult:
        """
        Clasifica una entidad como PII o Ruido.
        
        FLUJO SIMPLIFICADO:
        1. (Opcional) Detectar PII obvio por regex
        2. Clasificar con SetFit
        3. (Opcional) Filtrar por umbral de confianza
        """
        self.stats['total_processed'] += 1
        
        # PASO 1: Detector de PII obvio (opcional)
        if self.enable_pii_detector:
            is_pii, pii_pattern = self._is_obvious_pii(entity_text, entity_label)
            if is_pii:
                self.stats['pii_detected_regex'] += 1
                return ClassificationResult(
                    is_pii=True,
                    confidence=1.0,
                    classification_method="obvious_pii",
                    original_label=entity_label,
                    entity_text=entity_text,
                    details={"pattern": pii_pattern}
                )
        
        # PASO 2: Clasificación con SetFit
        input_text = f"ENTITY: {entity_text}\nSENTENCE: {sentence_context}"
        prediction, confidence = self._classify_with_setfit(input_text)
        
        # PASO 3: Filtro de baja confianza (opcional)
        if self.enable_low_confidence_filter:
            if prediction == 1 and confidence < self.confidence_threshold:
                self.stats['low_confidence_filtered'] += 1
                return ClassificationResult(
                    is_pii=False,
                    confidence=confidence,
                    classification_method="low_confidence",
                    original_label=entity_label,
                    entity_text=entity_text,
                    details={
                        "setfit_prediction": prediction,
                        "threshold": self.confidence_threshold
                    }
                )
        
        # PASO 4: Resultado final
        is_pii = prediction == 1
        if is_pii:
            self.stats['setfit_pii'] += 1
        else:
            self.stats['setfit_noise'] += 1
        
        return ClassificationResult(
            is_pii=is_pii,
            confidence=confidence,
            classification_method="setfit",
            original_label=entity_label,
            entity_text=entity_text,
            details={"raw_prediction": prediction}
        )
    
    def classify_batch(
        self,
        entities: List[Dict[str, Any]],
        full_document: Optional[str] = None,
    ) -> List[ClassificationResult]:
        """Clasifica un lote de entidades."""
        all_entity_texts = [e.get('entity_text', e.get('text', '')) for e in entities]
        
        results = []
        for entity in entities:
            entity_text = entity.get('entity_text', entity.get('text', ''))
            entity_label = entity.get('entity_label', entity.get('label', ''))
            context = entity.get('sentence_context', entity.get('context', ''))
            
            result = self.classify(
                entity_text=entity_text,
                entity_label=entity_label,
                sentence_context=context,
                full_document=full_document,
                other_entities=all_entity_texts,
            )
            results.append(result)
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Devuelve estadísticas de clasificación."""
        total = self.stats['total_processed']
        if total == 0:
            return self.stats
        
        return {
            **self.stats,
            'percentages': {
                'pii_detected_regex': self.stats['pii_detected_regex'] / total * 100,
                'low_confidence_filtered': self.stats['low_confidence_filtered'] / total * 100,
                'setfit_pii': self.stats['setfit_pii'] / total * 100,
                'setfit_noise': self.stats['setfit_noise'] / total * 100,
            }
        }
    
    def reset_statistics(self):
        """Reinicia las estadísticas."""
        for key in self.stats:
            self.stats[key] = 0


def create_gatekeeper(
    model_path: str = "models/gatekeeper_setfit",
    threshold: float = 0.85
) -> SetFitGatekeeper:
    """
    Crea una instancia del gatekeeper con configuración por defecto.
    """
    return SetFitGatekeeper(
        model_path=model_path,
        confidence_threshold=threshold,
        enable_pii_detector=False,
        enable_low_confidence_filter=True,
    )
