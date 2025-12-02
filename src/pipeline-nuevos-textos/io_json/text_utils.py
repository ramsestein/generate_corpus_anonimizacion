"""
Utilidad de normalización de texto para evaluaciones.

Función compartida por todos los módulos de evaluación para garantizar
comparación consistente de textos contra ground truth.
"""

import unicodedata


def normalize_text(text: str) -> str:
    """
    Normaliza texto para comparación: minúsculas, sin tildes, espacios normalizados.
    
    Args:
        text: Texto a normalizar
        
    Returns:
        Texto normalizado
    """
    # Minúsculas
    text = text.lower().strip()
    
    # Quit

ar tildes (NFD decomposition)
    text = ''.join(
        c for c in unicodedata.normalize('NFD', text)
        if unicodedata.category(c) != 'Mn'
    )
    
    # Normalizar espacios múltiples
    text = ' '.join(text.split())
    
    return text
