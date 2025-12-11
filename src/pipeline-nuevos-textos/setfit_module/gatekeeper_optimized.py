#!/usr/bin/env python3
"""
SetFit Gatekeeper - Clasificador principal PII vs Ruido (OPTIMIZADO)
=====================================================================

OPTIMIZACIONES APLICADAS:
- Patrones regex precompilados (compilación una sola vez)
- Caché de modelo SetFit (singleton global)
- Sets para búsquedas O(1) en lugar de listas
- Eliminación de normalizaciones redundantes
- Mejor estructura de datos para patrones

Sin cambios semánticos: sigue siendo el mismo clasificador binario PII vs ruido.
"""

import re
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Set
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# ============================================================================
# OPTIMIZACIÓN 1: Compilar patrones regex una sola vez (module-level)
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

# OPTIMIZACIÓN 2: Usar sets en lugar de listas (búsqueda O(1))
COMMON_WORDS = {
    'paciente', 'hospital', 'servicio', 'unidad', 'centro', 'clinica', 'consulta',
    'medico', 'enfermera', 'enfermero', 'doctor', 'doctora', 'cirujano',
    'españa', 'madrid', 'barcelona', 'valencia', 'sevilla', 'bilbao',
    'nacional', 'general', 'universitario', 'regional', 'provincial',
    'urgencias', 'emergencias', 'consultas', 'externas', 'hospitalizacion',
    'medicina', 'cirugia', 'pediatria', 'cardiologia', 'neurologia',
    'tratamiento', 'diagnostico', 'evolucion', 'anamnesis', 'exploracion',
    'informe', 'historia', 'clinica', 'medica', 'alta', 'ingreso',
    'el', 'la', 'los', 'las', 'un', 'una', 'unos', 'unas',
    'del', 'de', 'en', 'con', 'por', 'para', 'ante', 'sobre',
    'este', 'esta', 'estos', 'estas', 'ese', 'esa', 'esos', 'esas',
    'i', 'ii', 'iii', 'iv', 'v', 'vi', 'vii', 'viii', 'ix', 'x',
    'primero', 'segundo', 'tercero', 'cuarto', 'quinto',
    'enero', 'febrero', 'marzo', 'abril', 'mayo', 'junio',
    'julio', 'agosto', 'septiembre', 'octubre', 'noviembre', 'diciembre',
    'lunes', 'martes', 'miercoles', 'jueves', 'viernes', 'sabado', 'domingo',
}

ISOLATED_TITLES = {'dr', 'dra', 'sr', 'sra', 'don', 'doña', 'dña', 'd', 'prof', 'lic'}

FRAGMENT_PATTERNS = {
    '+', 'b', 'c', 'hc', 'cp', 'tf', 'tel', 'nº', 'n°', 'num',
    'calle', 'c/', 'avda', 'av', 'pza', 'plaza',
}

# ============================================================================
# OPTIMIZACIÓN 3: Singleton global para el modelo SetFit
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
    Clasificador SetFit mejorado (OPTIMIZADO).
    
    Cambios principales:
    - Modelo cargado una sola vez (singleton)
    - Patrones regex precompilados
    - Sets para búsquedas O(1)
    - Menos normalizaciones redundantes
    """
    
    def __init__(
        self,
        model_path: str = "models/gatekeeper_setfit",
        confidence_threshold: float = 0.75,
        enable_noise_filter: bool = False,
        enable_pii_detector: bool = False,
        enable_fragment_filter: bool = False,
        enable_low_confidence_filter: bool = False,
    ):
        """Inicializa el gatekeeper con el modelo SetFit."""
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.enable_noise_filter = enable_noise_filter
        self.enable_pii_detector = enable_pii_detector
        self.enable_fragment_filter = enable_fragment_filter
        self.enable_low_confidence_filter = enable_low_confidence_filter
        
        # OPTIMIZACIÓN: Modelo cargado una sola vez (lazy loading via propiedad)
        self._model = None
        self._model_loaded = False
        
        # Estadísticas
        self.stats = {
            'total_processed': 0,
            'noise_filtered': 0,
            'pii_detected': 0,
            'fragments_filtered': 0,
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
    
    def _is_obvious_noise(self, entity_text: str, entity_label: str) -> Tuple[bool, str]:
        """Detecta ruido obvio (optimizado con sets)."""
        normalized = entity_text.lower().strip()
        
        # OPTIMIZACIÓN: Búsquedas en sets O(1)
        if normalized in COMMON_WORDS:
            return True, f"common_word:{normalized}"
        
        if normalized in ISOLATED_TITLES:
            return True, f"isolated_title:{normalized}"
        
        if len(entity_text.strip()) <= 2 and not entity_text.strip().isdigit():
            return True, f"too_short:{entity_text}"
        
        if re.match(r'^\d{1,2}$', entity_text.strip()):
            return True, f"single_number:{entity_text}"
        
        if re.match(r'^[^\w]+$', entity_text):
            return True, f"punctuation_only:{entity_text}"
        
        if normalized in FRAGMENT_PATTERNS:
            return True, f"known_fragment:{normalized}"
        
        return False, ""
    
    def _is_obvious_pii(self, entity_text: str, entity_label: str) -> Tuple[bool, str]:
        """Detecta PII obvio (optimizado con patrones precompilados)."""
        # OPTIMIZACIÓN: Usar patrones precompilados (compilados una sola vez)
        for pattern_name, pattern in COMPILED_PII_PATTERNS.items():
            if pattern.match(entity_text.strip()):
                return True, pattern_name
        
        # Detector de nombres compuestos (caso especial)
        if entity_label in ['NOMBRE_SUJETO_ASISTENCIA', 'NOMBRE_PERSONAL_SANITARIO']:
            words = entity_text.split()
            if len(words) >= 2 and all(w[0].isupper() for w in words if len(w) > 1):
                return True, "compound_name"
        
        return False, ""
    
    def _is_fragment(
        self,
        entity_text: str,
        entity_label: str,
        full_document: Optional[str] = None,
        other_entities: Optional[List[str]] = None
    ) -> Tuple[bool, str]:
        """Detecta fragmentos (optimizado)."""
        normalized = entity_text.lower().strip()
        
        if len(normalized) <= 2:
            return True, "too_short"
        
        # OPTIMIZACIÓN: Búsqueda directa en lista (sin normalización repetida)
        if other_entities:
            for other in other_entities:
                other_norm = other.lower().strip()
                if normalized != other_norm and normalized in other_norm:
                    return True, f"subset_of:{other}"
        
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
        """Clasifica una entidad como PII o Ruido (optimizado)."""
        self.stats['total_processed'] += 1
        
        # PASO 1: Filtro de ruido obvio
        if self.enable_noise_filter:
            is_noise, noise_reason = self._is_obvious_noise(entity_text, entity_label)
            if is_noise:
                self.stats['noise_filtered'] += 1
                return ClassificationResult(
                    is_pii=False,
                    confidence=1.0,
                    classification_method="obvious_noise",
                    original_label=entity_label,
                    entity_text=entity_text,
                    details={"reason": noise_reason}
                )
        
        # PASO 2: Detector de PII obvio
        if self.enable_pii_detector:
            is_pii, pii_pattern = self._is_obvious_pii(entity_text, entity_label)
            if is_pii:
                self.stats['pii_detected'] += 1
                return ClassificationResult(
                    is_pii=True,
                    confidence=1.0,
                    classification_method="obvious_pii",
                    original_label=entity_label,
                    entity_text=entity_text,
                    details={"pattern": pii_pattern}
                )
        
        # PASO 3: Filtro de fragmentos
        if self.enable_fragment_filter:
            is_fragment, fragment_parent = self._is_fragment(
                entity_text, entity_label, full_document, other_entities
            )
            if is_fragment:
                self.stats['fragments_filtered'] += 1
                return ClassificationResult(
                    is_pii=False,
                    confidence=1.0,
                    classification_method="fragment",
                    original_label=entity_label,
                    entity_text=entity_text,
                    details={"parent": fragment_parent}
                )
        
        # PASO 4: Clasificación con SetFit (optimizado: construir string una sola vez)
        input_text = f"ENTITY: {entity_text}\nSENTENCE: {sentence_context}"
        prediction, confidence = self._classify_with_setfit(input_text)
        
        # PASO 5: Filtro de baja confianza
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
        
        # PASO 6: Resultado final
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
        """Clasifica un lote de entidades (optimizado)."""
        # OPTIMIZACIÓN: Precalcular lista de textos una sola vez
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
                'noise_filtered': self.stats['noise_filtered'] / total * 100,
                'pii_detected': self.stats['pii_detected'] / total * 100,
                'fragments_filtered': self.stats['fragments_filtered'] / total * 100,
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
    threshold: float = 0.75
) -> SetFitGatekeeper:
    """
    Crea una instancia del gatekeeper con configuración por defecto.
    """
    return SetFitGatekeeper(
        model_path=model_path,
        confidence_threshold=threshold,
        enable_noise_filter=False,
        enable_pii_detector=False,
        enable_fragment_filter=False,
        enable_low_confidence_filter=False,
    )
