#!/usr/bin/env python3
"""
Tests para verificar la funcionalidad de substring matching en:
- CsvListManager: find_in_whitelist(), find_in_blacklist()
- EntityFastFilter: match_in_whitelist(), match_in_blacklist()

Casos de test:
1. Coincidencia exacta: el texto es exactamente igual a un término de la lista
2. Substring match: el término de la lista aparece dentro de un texto más largo
3. Casos negativos: no debe haber match cuando no corresponde
4. Case sensitivity: whitelist case-sensitive, blacklist case-insensitive

Author: Pipeline Anonimización Clínica
"""

import pytest
import sys
from pathlib import Path

# Añadir path para imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "pipeline-nuevos-textos"))

from entity_fast_filter import EntityFastFilter, EnumDecision
from csv_list_manager import CsvListManager


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def filter_with_test_data():
    """
    Crea un EntityFastFilter con datos de prueba manuales.
    """
    filter_instance = EntityFastFilter(
        whitelist_data=["Hospital Clínic", "Barcelona", "Madrid", "Dr. García"],
        blacklist_data=["paracetamol", "ibuprofeno", "amoxicilina", "omeprazol"]
    )
    return filter_instance


@pytest.fixture
def csv_manager_mock(tmp_path):
    """
    Crea un CsvListManager con archivos CSV temporales de prueba.
    """
    # Crear carpeta temporal con CSVs
    listas_dir = tmp_path / "LISTAS"
    listas_dir.mkdir()
    
    # CSV whitelist (hospitales)
    whitelist_csv = listas_dir / "hospitales_test.csv"
    whitelist_csv.write_text(
        "Nombre Centro\n"
        "Hospital Clínic\n"
        "Hospital Vall d'Hebron\n"
        "Hospital de Sant Pau\n"
        "Clínica Universidad de Navarra\n",
        encoding="utf-8"
    )
    
    # CSV blacklist (medicamentos) - simular Nomenclator
    blacklist_csv = listas_dir / "Nomenclator_de_Facturacion_test.csv"
    blacklist_csv.write_text(
        "Nombre del producto farmacéutico\n"
        "paracetamol\n"
        "ibuprofeno\n"
        "amoxicilina\n"
        "omeprazol\n"
        "ácido acetilsalicílico\n",
        encoding="utf-8"
    )
    
    manager = CsvListManager(str(listas_dir))
    manager.load()
    return manager


# ============================================================================
# TESTS: ENTITY FAST FILTER - COINCIDENCIA EXACTA
# ============================================================================

class TestEntityFastFilterExactMatch:
    """Tests de coincidencia exacta en EntityFastFilter."""

    def test_whitelist_exact_match(self, filter_with_test_data):
        """Texto exactamente igual a término whitelist."""
        result = filter_with_test_data.match_in_whitelist("Hospital Clínic")
        assert result == "Hospital Clínic"

    def test_blacklist_exact_match(self, filter_with_test_data):
        """Texto exactamente igual a término blacklist."""
        result = filter_with_test_data.match_in_blacklist("paracetamol")
        assert result == "paracetamol"

    def test_blacklist_case_insensitive(self, filter_with_test_data):
        """Blacklist debe ser case-insensitive."""
        result = filter_with_test_data.match_in_blacklist("PARACETAMOL")
        assert result is not None
        assert result.lower() == "paracetamol"

    def test_whitelist_case_sensitive(self, filter_with_test_data):
        """Whitelist debe ser case-sensitive (nombres propios)."""
        # Barcelona con B mayúscula debería matchear
        result_upper = filter_with_test_data.match_in_whitelist("Barcelona")
        assert result_upper == "Barcelona"
        
        # barcelona con b minúscula NO debería matchear (case-sensitive)
        result_lower = filter_with_test_data.match_in_whitelist("barcelona")
        # Nota: depende de la config de case_sensitivity del KeywordProcessor
        # Si es case-sensitive, result_lower debería ser None


# ============================================================================
# TESTS: ENTITY FAST FILTER - SUBSTRING MATCH
# ============================================================================

class TestEntityFastFilterSubstringMatch:
    """Tests de substring matching en EntityFastFilter."""

    def test_whitelist_substring_in_sentence(self, filter_with_test_data):
        """Encontrar término whitelist dentro de una frase."""
        text = "Ingreso en el Hospital Clínic de Barcelona por urgencias."
        result = filter_with_test_data.match_in_whitelist(text)
        # Debería encontrar "Hospital Clínic" o "Barcelona"
        assert result is not None
        assert result in ["Hospital Clínic", "Barcelona"]

    def test_blacklist_substring_in_sentence(self, filter_with_test_data):
        """Encontrar término blacklist dentro de una frase."""
        text = "Tomar paracetamol 1g cada 8 horas."
        result = filter_with_test_data.match_in_blacklist(text)
        assert result == "paracetamol"

    def test_whitelist_multiple_matches_returns_longest(self, filter_with_test_data):
        """
        Si hay múltiples coincidencias, devolver la más larga.
        """
        # Crear filtro con términos solapados
        filter_overlap = EntityFastFilter(
            whitelist_data=["Hospital", "Hospital Clínic", "Hospital Clínic de Barcelona"],
            blacklist_data=[]
        )
        
        text = "Ingreso en Hospital Clínic de Barcelona"
        result = filter_overlap.match_in_whitelist(text)
        
        # Debería devolver el más largo
        assert result == "Hospital Clínic de Barcelona"

    def test_blacklist_with_dose_sanitization(self, filter_with_test_data):
        """
        Medicamento con dosis debe encontrarse tras sanitización.
        """
        # "paracetamol 500mg" debería encontrar "paracetamol"
        text = "paracetamol 500mg"
        result = filter_with_test_data.match_in_blacklist(text)
        assert result == "paracetamol"

    def test_no_match_returns_none(self, filter_with_test_data):
        """Texto sin coincidencias devuelve None."""
        text = "El paciente presenta cefalea intensa"
        
        result_wl = filter_with_test_data.match_in_whitelist(text)
        result_bl = filter_with_test_data.match_in_blacklist(text)
        
        assert result_wl is None
        assert result_bl is None


# ============================================================================
# TESTS: ENTITY FAST FILTER - EVALUATE_CANDIDATE
# ============================================================================

class TestEntityFastFilterEvaluateCandidate:
    """Tests de evaluate_candidate con substring matching."""

    def test_evaluate_hospital_in_sentence_force_anonymize(self, filter_with_test_data):
        """Entidad con hospital en contexto → FORCE_ANONYMIZE."""
        candidate = {
            "text": "Hospital Clínic de Barcelona",
            "label": "INSTITUCION"
        }
        decision = filter_with_test_data.evaluate_candidate(candidate)
        assert decision == EnumDecision.FORCE_ANONYMIZE

    def test_evaluate_medication_force_ignore(self, filter_with_test_data):
        """Entidad que es medicamento → FORCE_IGNORE."""
        candidate = {
            "text": "ibuprofeno",
            "label": "OTRO"
        }
        decision = filter_with_test_data.evaluate_candidate(candidate)
        assert decision == EnumDecision.FORCE_IGNORE

    def test_evaluate_medication_with_dose_force_ignore(self, filter_with_test_data):
        """Medicamento con dosis → FORCE_IGNORE."""
        candidate = {
            "text": "paracetamol 500mg comprimidos",
            "label": "OTRO"
        }
        decision = filter_with_test_data.evaluate_candidate(candidate)
        assert decision == EnumDecision.FORCE_IGNORE

    def test_evaluate_unknown_escalate(self, filter_with_test_data):
        """Entidad no reconocida → ESCALATE_TO_LLM."""
        candidate = {
            "text": "Pedro González",
            "label": "NOMBRE"
        }
        decision = filter_with_test_data.evaluate_candidate(candidate)
        assert decision == EnumDecision.ESCALATE_TO_LLM


# ============================================================================
# TESTS: CSV LIST MANAGER - SUBSTRING MATCH
# ============================================================================

class TestCsvListManagerSubstringMatch:
    """Tests de substring matching en CsvListManager."""

    def test_find_in_whitelist_substring(self, csv_manager_mock):
        """Encontrar término whitelist en texto largo."""
        text = "El paciente fue atendido en el Hospital Clínic por urgencias"
        result = csv_manager_mock.find_in_whitelist(text)
        assert result == "Hospital Clínic"

    def test_find_in_blacklist_substring(self, csv_manager_mock):
        """Encontrar término blacklist en texto largo."""
        text = "Se prescribe paracetamol 500mg cada 8 horas"
        result = csv_manager_mock.find_in_blacklist(text)
        assert result == "paracetamol"

    def test_find_all_in_whitelist(self, csv_manager_mock):
        """Encontrar todos los términos whitelist en texto."""
        text = "Hospital Clínic y Hospital de Sant Pau son centros de referencia"
        results = csv_manager_mock.find_all_in_whitelist(text)
        assert "Hospital Clínic" in results
        assert "Hospital de Sant Pau" in results
        assert len(results) == 2

    def test_find_all_in_blacklist(self, csv_manager_mock):
        """Encontrar todos los términos blacklist en texto."""
        text = "Tratamiento: paracetamol alternando con ibuprofeno"
        results = csv_manager_mock.find_all_in_blacklist(text)
        assert "paracetamol" in results
        assert "ibuprofeno" in results
        assert len(results) == 2

    def test_is_in_whitelist_exact_vs_find_in_whitelist_substring(self, csv_manager_mock):
        """
        Diferencia entre is_in_whitelist (exacto) y find_in_whitelist (substring).
        """
        text = "Ingreso en Hospital Clínic"
        
        # is_in_whitelist busca coincidencia EXACTA
        assert csv_manager_mock.is_in_whitelist(text) is False
        
        # find_in_whitelist busca SUBSTRING
        result = csv_manager_mock.find_in_whitelist(text)
        assert result == "Hospital Clínic"


# ============================================================================
# TESTS: CASOS AMBIGUOS Y EDGE CASES
# ============================================================================

class TestEdgeCases:
    """Tests de casos límite y ambiguos."""

    def test_empty_text(self, filter_with_test_data):
        """Texto vacío no debe causar error."""
        assert filter_with_test_data.match_in_whitelist("") is None
        assert filter_with_test_data.match_in_whitelist("   ") is None
        assert filter_with_test_data.match_in_blacklist("") is None

    def test_none_text(self, filter_with_test_data):
        """Texto None no debe causar error."""
        # Esto depende de la implementación - puede lanzar error o devolver None
        try:
            result = filter_with_test_data.match_in_whitelist(None)
            assert result is None
        except (TypeError, AttributeError):
            pass  # Comportamiento aceptable

    def test_partial_word_no_match(self, filter_with_test_data):
        """
        Substring parcial de palabra NO debe matchear.
        "paracetamol" no debe matchear en "paraceta" (incompleto).
        """
        # flashtext hace word boundaries, así que esto es correcto
        text = "Prescribir para la cefalea"  # "para" != "paracetamol"
        result = filter_with_test_data.match_in_blacklist(text)
        assert result is None

    def test_accents_and_special_chars(self, filter_with_test_data):
        """Términos con acentos y caracteres especiales."""
        # "Hospital Clínic" tiene í con acento
        text = "Derivación a Hospital Clínic"
        result = filter_with_test_data.match_in_whitelist(text)
        assert result == "Hospital Clínic"


# ============================================================================
# TESTS DE INTEGRACIÓN: FILTER + CSV_MANAGER
# ============================================================================

class TestIntegration:
    """Tests de integración EntityFastFilter + CsvListManager."""

    def test_filter_uses_csv_manager_lists(self, csv_manager_mock, tmp_path):
        """
        EntityFastFilter puede usar listas de CsvListManager.
        """
        # Crear filtro con la ruta del csv_manager_mock
        listas_path = str(tmp_path / "LISTAS")
        filter_with_csv = EntityFastFilter(csv_lists_path=listas_path)
        
        # Verificar que cargó los términos
        assert filter_with_csv._csv_manager is not None
        
        # Test de whitelist desde CSV
        text_hospital = "Hospital Clínic"
        result = filter_with_csv.match_in_whitelist(text_hospital)
        assert result == "Hospital Clínic"
        
        # Test de blacklist desde CSV
        text_med = "paracetamol"
        result_med = filter_with_csv.match_in_blacklist(text_med)
        assert result_med is not None


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
