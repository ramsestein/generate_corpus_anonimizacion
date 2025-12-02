#!/usr/bin/env python3
"""
SetFit Gatekeeper - Clasificador principal PII vs Ruido
========================================================

Clasificador SetFit mejorado con filtros pre y post clasificación.
Mantiene la arquitectura: MEDDOCAN → SetFit → listas → LLM

Mejoras implementadas:
1. Filtro de ruido obvio (pre-SetFit)
2. Detector de PII obvio (bypass SetFit)
3. Filtro de fragmentos
4. Umbral de confianza ajustable (default: 0.75)
"""

import re
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field

# Configurar logging
logger = logging.getLogger(__name__)


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
    Clasificador SetFit mejorado con filtros pre y post clasificación.
    
    El gatekeeper mantiene la arquitectura del pipeline pero añade:
    - Filtro de ruido obvio antes de SetFit
    - Detector de PII obvio (bypass SetFit)
    - Detección de fragmentos de entidades
    - Umbral de confianza ajustable
    """
    
    # =========================================================================
    # PATRONES DE RUIDO OBVIO
    # =========================================================================
    
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
    
    # =========================================================================
    # PATRONES DE PII OBVIO
    # =========================================================================
    
    PII_PATTERNS = {
        'email': r'^[\w.+-]+@[\w-]+\.[\w.-]+$',
        'phone_es': r'^(\+34\s?)?(6|7|8|9)\d{2}[\s.-]?\d{3}[\s.-]?\d{3}$',
        'phone_intl': r'^\+\d{1,3}[\s.-]?\d{3,4}[\s.-]?\d{3,4}[\s.-]?\d{3,4}$',
        'dni_nif': r'^[0-9]{8}[A-Z]$',
        'nie': r'^[XYZ][0-9]{7}[A-Z]$',
        'iban_es': r'^ES\d{22}$',
        'credit_card': r'^\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}$',
        'ss_number': r'^\d{12}$',
        'postal_code': r'^\d{5}$',
        'license_plate': r'^[0-9]{4}[A-Z]{3}$|^[A-Z]{1,2}[0-9]{4}[A-Z]{2}$',
        'full_date': r'^\d{1,2}[/-]\d{1,2}[/-]\d{2,4}$',
    }
    
    def __init__(
        self,
        model_path: str = "models/gatekeeper_setfit",
        confidence_threshold: float = 0.75,
        enable_noise_filter: bool = False,
        enable_pii_detector: bool = False,
        enable_fragment_filter: bool = False,
        enable_low_confidence_filter: bool = False,
    ):
        """
        Inicializa el gatekeeper con el modelo SetFit.
        
        Args:
            model_path: Ruta al modelo SetFit entrenado
            confidence_threshold: Umbral mínimo de confianza (0.0-1.0)
            enable_noise_filter: Activar filtro de ruido obvio
            enable_pii_detector: Activar detector de PII obvio
            enable_fragment_filter: Activar filtro de fragmentos
            enable_low_confidence_filter: Activar filtro de baja confianza
        """
        self.model_path = Path(model_path)
        self.confidence_threshold = confidence_threshold
        self.enable_noise_filter = enable_noise_filter
        self.enable_pii_detector = enable_pii_detector
        self.enable_fragment_filter = enable_fragment_filter
        self.enable_low_confidence_filter = enable_low_confidence_filter
        
        # Modelo (carga diferida)
        self._model = None
        
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
        
        logger.info(f"SetFitGatekeeper configurado con umbral={confidence_threshold}")
    
    @property
    def model(self):
        """Carga diferida del modelo SetFit."""
        if self._model is None:
            self._model = self._load_model()
        return self._model
    
    def _load_model(self):
        """Carga el modelo SetFit."""
        try:
            from setfit import SetFitModel
        except ImportError:
            raise ImportError(
                "SetFit no está instalado. Ejecuta: pip install setfit"
            )
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Modelo no encontrado: {self.model_path}")
        
        logger.info(f"Cargando modelo SetFit desde {self.model_path}")
        return SetFitModel.from_pretrained(str(self.model_path))
    
    def _normalize_text(self, text: str) -> str:
        """Normaliza texto para comparaciones."""
        return text.lower().strip()
    
    def _is_obvious_noise(self, entity_text: str, entity_label: str) -> Tuple[bool, str]:
        """Detecta si una entidad es ruido obvio."""
        normalized = self._normalize_text(entity_text)
        
        if normalized in self.COMMON_WORDS:
            return True, f"common_word:{normalized}"
        
        if normalized in self.ISOLATED_TITLES:
            return True, f"isolated_title:{normalized}"
        
        if len(entity_text.strip()) <= 2 and not entity_text.strip().isdigit():
            return True, f"too_short:{entity_text}"
        
        if re.match(r'^\d{1,2}$', entity_text.strip()):
            return True, f"single_number:{entity_text}"
        
        if re.match(r'^[^\w]+$', entity_text):
            return True, f"punctuation_only:{entity_text}"
        
        if normalized in self.FRAGMENT_PATTERNS:
            return True, f"known_fragment:{normalized}"
        
        if normalized in {'el', 'la', 'los', 'las', 'de', 'del', 'en', 'con', 'por'}:
            return True, f"article_preposition:{normalized}"
        
        return False, ""
    
    def _is_obvious_pii(self, entity_text: str, entity_label: str) -> Tuple[bool, str]:
        """Detecta si una entidad es PII obvio."""
        for pattern_name, pattern in self.PII_PATTERNS.items():
            if re.match(pattern, entity_text.strip(), re.IGNORECASE):
                return True, pattern_name
        
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
        """Detecta si una entidad es un fragmento de otra entidad más larga."""
        normalized = self._normalize_text(entity_text)
        
        if len(normalized) <= 2:
            return True, "too_short"
        
        if other_entities:
            for other in other_entities:
                other_norm = self._normalize_text(other)
                if normalized != other_norm and normalized in other_norm:
                    return True, f"subset_of:{other}"
        
        if full_document:
            pattern = rf'\b\w+\s+{re.escape(entity_text)}\s+\w+\b'
            matches = re.findall(pattern, full_document, re.IGNORECASE)
            if matches:
                return True, f"part_of:{matches[0]}"
        
        return False, ""
    
    def _build_input_text(
        self,
        entity_text: str,
        sentence_context: str,
        entity_label: Optional[str] = None
    ) -> str:
        """Construye el texto de entrada para SetFit."""
        return f"ENTITY: {entity_text}\nSENTENCE: {sentence_context}"
    
    def _classify_with_setfit(self, input_text: str) -> Tuple[int, float]:
        """Clasifica usando el modelo SetFit."""
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
        
        Args:
            entity_text: Texto de la entidad
            entity_label: Etiqueta de la entidad (de MEDDOCAN)
            sentence_context: Contexto de la oración
            full_document: Documento completo (opcional)
            other_entities: Otras entidades del documento (opcional)
        
        Returns:
            ClassificationResult con la decisión y metadatos
        """
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
        
        # PASO 4: Clasificación con SetFit
        input_text = self._build_input_text(entity_text, sentence_context, entity_label)
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
    
    Args:
        model_path: Ruta al modelo SetFit
        threshold: Umbral de confianza (recomendado: 0.75)
    
    Returns:
        SetFitGatekeeper configurado
    """
    return SetFitGatekeeper(
        model_path=model_path,
        confidence_threshold=threshold,
        enable_noise_filter=False,
        enable_pii_detector=False,
        enable_fragment_filter=False,
        enable_low_confidence_filter=False,
    )
