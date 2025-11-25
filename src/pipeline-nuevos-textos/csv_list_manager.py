#!/usr/bin/env python3
"""
CsvListManager: Gestor de listas blancas/negras desde archivos CSV/Excel.

Este módulo carga listas de términos desde archivos CSV/XLS/XLSX en una carpeta
y las clasifica automáticamente como whitelist o blacklist según el nombre del archivo.

Convención de nomenclatura:
    - Archivos con "Nomenclator_de_Facturacion" en el nombre → BLACKLIST (medicamentos)
    - Resto de archivos → WHITELIST (hospitales, lugares, etc.)

Columnas conocidas por tipo de archivo:
    - CNH (hospitales): "Nombre Centro", "Municipio", "Provincia"
    - Nomenclator (medicamentos): "Nombre del producto farmacéutico", "Principio activo..."
    - codmun (municipios): "Provincia" (detectada automáticamente)

Author: Pipeline Anonimización Clínica
Version: 1.0.0
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd
from flashtext import KeywordProcessor

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURACIÓN DE COLUMNAS CONOCIDAS
# ============================================================================

# Mapeo de patrones de nombre de archivo a columnas relevantes
KNOWN_COLUMN_MAPPINGS: Dict[str, List[str]] = {
    # CNH (Catálogo Nacional de Hospitales)
    "CNH": [
        "Nombre Centro",
        "Municipio", 
        "Provincia",
        "CCAA",
        "Nombre del Complejo"
    ],
    # Nomenclator de Facturación (medicamentos - blacklist)
    "Nomenclator_de_Facturacion": [
        "Nombre del producto farmacéutico",
        "Principio activo o asociación de principios activos",
        "Nombre genérico efecto y accesorio",
        "Nombre de la agrupación homogénea del producto sanitario"
    ],
    # Códigos de municipios
    "codmun": [
        "Provincia"
    ]
}

# Columnas genéricas que podrían contener valores relevantes
GENERIC_VALUE_COLUMNS = [
    "value", "nombre", "name", "descripcion", "description",
    "texto", "text", "termino", "term", "entidad", "entity"
]


# ============================================================================
# CLASE PRINCIPAL
# ============================================================================

class CsvListManager:
    """
    Gestor de listas blancas y negras desde archivos CSV/Excel.

    Carga automáticamente todos los archivos CSV/XLS/XLSX de una carpeta,
    los clasifica como whitelist o blacklist según su nombre, y proporciona
    métodos para consultar si un texto está en alguna lista.

    Attributes:
        base_path: Ruta a la carpeta con los archivos.
        whitelists: Dict con listas blancas por archivo.
        blacklists: Dict con listas negras por archivo.
        whitelist_global: Set con todos los términos de whitelists.
        blacklist_global: Set con todos los términos de blacklists.

    Example:
        >>> manager = CsvListManager("LISTAS/")
        >>> manager.load()
        >>> manager.is_in_whitelist("Hospital Clínic")
        True
        >>> manager.is_in_blacklist("Paracetamol")
        True
    """

    # Patrón para identificar blacklists
    BLACKLIST_PATTERN = re.compile(r"Nomenclator_de_Facturacion", re.IGNORECASE)

    # Extensiones soportadas
    SUPPORTED_EXTENSIONS = {".csv", ".xls", ".xlsx"}

    def __init__(
        self,
        base_path: str = "LISTAS/",
        lowercase_whitelist: bool = False,
        lowercase_blacklist: bool = True
    ) -> None:
        """
        Inicializa el gestor de listas.

        Args:
            base_path: Ruta a la carpeta con archivos CSV/Excel.
            lowercase_whitelist: Si True, normaliza whitelist a minúsculas.
                                 Default False (case-sensitive para nombres propios).
            lowercase_blacklist: Si True, normaliza blacklist a minúsculas.
                                 Default True (medicamentos case-insensitive).
        """
        self.base_path = Path(base_path)
        self.lowercase_whitelist = lowercase_whitelist
        self.lowercase_blacklist = lowercase_blacklist

        # Estructuras de datos
        self.whitelists: Dict[str, List[str]] = {}
        self.blacklists: Dict[str, List[str]] = {}
        self.whitelist_global: Set[str] = set()
        self.blacklist_global: Set[str] = set()

        # Procesadores flashtext para búsqueda de substrings O(N)
        # Se inicializan después de cargar los datos
        self._whitelist_processor: Optional[KeywordProcessor] = None
        self._blacklist_processor: Optional[KeywordProcessor] = None

        # Estadísticas
        self._stats: Dict[str, Any] = {
            "files_processed": 0,
            "files_failed": 0,
            "whitelist_files": 0,
            "blacklist_files": 0
        }

        self._loaded = False

    def load(self) -> None:
        """
        Carga todos los archivos CSV/Excel de la carpeta base.

        Clasifica cada archivo como whitelist o blacklist según su nombre,
        extrae los valores relevantes y los almacena en las estructuras internas.

        Raises:
            FileNotFoundError: Si la carpeta base no existe.
        """
        if not self.base_path.exists():
            raise FileNotFoundError(f"Carpeta no encontrada: {self.base_path}")

        logger.info(f"Cargando listas desde: {self.base_path}")

        # Resetear estructuras
        self.whitelists.clear()
        self.blacklists.clear()
        self.whitelist_global.clear()
        self.blacklist_global.clear()
        self._stats = {
            "files_processed": 0,
            "files_failed": 0,
            "whitelist_files": 0,
            "blacklist_files": 0
        }

        # Procesar cada archivo
        for file_path in self.base_path.iterdir():
            if not file_path.is_file():
                continue

            if file_path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
                continue

            try:
                self._process_file(file_path)
                self._stats["files_processed"] += 1
            except Exception as e:
                logger.warning(f"Error procesando {file_path.name}: {e}")
                self._stats["files_failed"] += 1

        self._loaded = True

        # Inicializar procesadores flashtext para búsqueda de substrings
        self._init_flashtext_processors()

        # Log resumen
        logger.info(
            f"Carga completada: "
            f"{self._stats['files_processed']} archivos procesados, "
            f"{self._stats['files_failed']} fallidos. "
            f"Whitelist: {len(self.whitelist_global)} términos. "
            f"Blacklist: {len(self.blacklist_global)} términos."
        )

    def _process_file(self, file_path: Path) -> None:
        """
        Procesa un archivo individual y extrae sus valores.

        Args:
            file_path: Ruta al archivo CSV/Excel.
        """
        file_name = file_path.stem  # Nombre sin extensión
        is_blacklist = bool(self.BLACKLIST_PATTERN.search(file_name))

        logger.debug(f"Procesando: {file_path.name} ({'blacklist' if is_blacklist else 'whitelist'})")

        # Leer archivo
        df = self._read_file(file_path)
        if df is None or df.empty:
            logger.warning(f"Archivo vacío o no legible: {file_path.name}")
            return

        # Detectar columnas relevantes
        columns = self._detect_relevant_columns(file_name, df)
        if not columns:
            logger.warning(f"No se encontraron columnas relevantes en: {file_path.name}")
            return

        # Extraer valores
        values = self._extract_values(df, columns, lowercase=self.lowercase_blacklist if is_blacklist else self.lowercase_whitelist)

        if not values:
            logger.warning(f"No se extrajeron valores de: {file_path.name}")
            return

        # Almacenar según tipo
        if is_blacklist:
            self.blacklists[file_name] = values
            self.blacklist_global.update(values)
            self._stats["blacklist_files"] += 1
            logger.info(f"  ✓ Blacklist '{file_name}': {len(values)} términos")
        else:
            self.whitelists[file_name] = values
            self.whitelist_global.update(values)
            self._stats["whitelist_files"] += 1
            logger.info(f"  ✓ Whitelist '{file_name}': {len(values)} términos")

    def _init_flashtext_processors(self) -> None:
        """
        Inicializa los procesadores flashtext para búsqueda de substrings O(N).

        Los procesadores permiten buscar si algún término de la lista aparece
        como substring dentro de un texto más largo.

        - Whitelist: case-sensitive (según lowercase_whitelist)
        - Blacklist: case-insensitive (según lowercase_blacklist)
        """
        # Procesador whitelist
        self._whitelist_processor = KeywordProcessor(case_sensitive=not self.lowercase_whitelist)
        for term in self.whitelist_global:
            self._whitelist_processor.add_keyword(term)

        # Procesador blacklist
        self._blacklist_processor = KeywordProcessor(case_sensitive=not self.lowercase_blacklist)
        for term in self.blacklist_global:
            self._blacklist_processor.add_keyword(term)

        logger.debug(
            f"Procesadores flashtext inicializados: "
            f"whitelist={len(self.whitelist_global)}, "
            f"blacklist={len(self.blacklist_global)}"
        )

    def _read_file(self, file_path: Path) -> Optional[pd.DataFrame]:
        """
        Lee un archivo CSV o Excel.

        Args:
            file_path: Ruta al archivo.

        Returns:
            DataFrame con los datos o None si hay error.
        """
        try:
            suffix = file_path.suffix.lower()

            if suffix == ".csv":
                # Intentar diferentes encodings
                for encoding in ["utf-8", "latin-1", "cp1252"]:
                    try:
                        return pd.read_csv(file_path, encoding=encoding)
                    except UnicodeDecodeError:
                        continue
                return None

            elif suffix in (".xls", ".xlsx"):
                # Excel
                engine = "xlrd" if suffix == ".xls" else "openpyxl"
                return pd.read_excel(file_path, engine=engine)

        except Exception as e:
            logger.error(f"Error leyendo {file_path.name}: {e}")
            return None

    def _detect_relevant_columns(
        self,
        file_name: str,
        df: pd.DataFrame
    ) -> List[str]:
        """
        Detecta las columnas relevantes de un DataFrame.

        Estrategia:
        1. Buscar patrón conocido en nombre de archivo.
        2. Buscar columnas con nombres genéricos.
        3. Usar primera columna no vacía como fallback.

        Args:
            file_name: Nombre del archivo (sin extensión).
            df: DataFrame con los datos.

        Returns:
            Lista de nombres de columnas relevantes.
        """
        columns_found: List[str] = []
        df_columns = set(df.columns.tolist())

        # 1. Buscar patrón conocido
        for pattern, known_cols in KNOWN_COLUMN_MAPPINGS.items():
            if pattern.lower() in file_name.lower():
                for col in known_cols:
                    if col in df_columns:
                        columns_found.append(col)

        if columns_found:
            return columns_found

        # 2. Buscar columnas genéricas
        for generic_col in GENERIC_VALUE_COLUMNS:
            for df_col in df_columns:
                if generic_col.lower() in df_col.lower():
                    columns_found.append(df_col)

        if columns_found:
            return columns_found

        # 3. Fallback: primera columna con datos de texto
        for col in df.columns:
            if df[col].dtype == object:  # Columna de strings
                non_null = df[col].dropna()
                if len(non_null) > 0:
                    # Verificar que tiene strings, no números como strings
                    sample = non_null.head(10)
                    if any(isinstance(v, str) and not v.replace('.', '').replace(',', '').isdigit() 
                           for v in sample):
                        columns_found.append(col)
                        break

        return columns_found

    def _extract_values(
        self,
        df: pd.DataFrame,
        columns: List[str],
        lowercase: bool = False
    ) -> List[str]:
        """
        Extrae y normaliza valores de las columnas especificadas.

        Args:
            df: DataFrame con los datos.
            columns: Lista de columnas a extraer.
            lowercase: Si True, convierte a minúsculas.

        Returns:
            Lista de valores únicos normalizados.
        """
        values: Set[str] = set()

        for col in columns:
            if col not in df.columns:
                continue

            for val in df[col].dropna():
                if not isinstance(val, str):
                    val = str(val)

                # Normalizar
                val = val.strip()

                # Filtrar valores vacíos o muy cortos
                if len(val) < 2:
                    continue

                # Filtrar valores que parecen códigos numéricos
                if val.replace('.', '').replace(',', '').replace('-', '').isdigit():
                    continue

                # Aplicar lowercase si corresponde
                if lowercase:
                    val = val.lower()

                values.add(val)

        return list(values)

    def is_in_whitelist(self, text: str) -> bool:
        """
        Verifica si un texto está en alguna whitelist.

        Args:
            text: Texto a buscar.

        Returns:
            True si el texto está en la whitelist global.
        """
        if not self._loaded:
            logger.warning("Listas no cargadas. Llama a load() primero.")
            return False

        if not text or not text.strip():
            return False

        text_clean = text.strip()

        # Búsqueda exacta (whitelist es case-sensitive por defecto)
        if text_clean in self.whitelist_global:
            return True

        # Si whitelist está en lowercase, buscar en lowercase
        if self.lowercase_whitelist:
            return text_clean.lower() in self.whitelist_global

        return False

    def is_in_blacklist(self, text: str) -> bool:
        """
        Verifica si un texto está en alguna blacklist.

        Args:
            text: Texto a buscar.

        Returns:
            True si el texto está en la blacklist global.
        """
        if not self._loaded:
            logger.warning("Listas no cargadas. Llama a load() primero.")
            return False

        if not text or not text.strip():
            return False

        text_clean = text.strip()

        # Blacklist es case-insensitive por defecto
        if self.lowercase_blacklist:
            return text_clean.lower() in self.blacklist_global
        else:
            return text_clean in self.blacklist_global

    def get_whitelist_match(self, text: str) -> Optional[str]:
        """
        Busca coincidencia en whitelist y devuelve el término coincidente.

        Args:
            text: Texto a buscar.

        Returns:
            Término coincidente o None.
        """
        if self.is_in_whitelist(text):
            text_clean = text.strip()
            if self.lowercase_whitelist:
                return text_clean.lower()
            return text_clean
        return None

    def get_blacklist_match(self, text: str) -> Optional[str]:
        """
        Busca coincidencia en blacklist y devuelve el término coincidente.

        Args:
            text: Texto a buscar.

        Returns:
            Término coincidente o None.
        """
        if self.is_in_blacklist(text):
            text_clean = text.strip()
            if self.lowercase_blacklist:
                return text_clean.lower()
            return text_clean
        return None

    def find_in_whitelist(self, text: str) -> Optional[str]:
        """
        Busca si algún término de la whitelist aparece DENTRO del texto dado.

        Usa flashtext (Aho-Corasick) para búsqueda O(N) de substrings.
        Si hay múltiples coincidencias, devuelve la más larga.

        Args:
            text: Texto donde buscar (puede ser una frase o párrafo).

        Returns:
            El término de whitelist encontrado (el más largo si hay varios),
            o None si no hay coincidencia.

        Example:
            >>> manager.find_in_whitelist("Ingreso en el Hospital Clínic de Barcelona")
            "Hospital Clínic"  # Si "Hospital Clínic" está en whitelist
        """
        if not self._loaded or self._whitelist_processor is None:
            logger.warning("Listas no cargadas. Llama a load() primero.")
            return None

        if not text or not text.strip():
            return None

        matches = self._whitelist_processor.extract_keywords(text)

        if not matches:
            return None

        # Devolver el match más largo (más específico)
        return max(matches, key=len)

    def find_in_blacklist(self, text: str) -> Optional[str]:
        """
        Busca si algún término de la blacklist aparece DENTRO del texto dado.

        Usa flashtext (Aho-Corasick) para búsqueda O(N) de substrings.
        Si hay múltiples coincidencias, devuelve la más larga.

        Args:
            text: Texto donde buscar (puede ser una frase o párrafo).

        Returns:
            El término de blacklist encontrado (el más largo si hay varios),
            o None si no hay coincidencia.

        Example:
            >>> manager.find_in_blacklist("Toma paracetamol 500mg cada 8 horas")
            "paracetamol"  # Si "paracetamol" está en blacklist
        """
        if not self._loaded or self._blacklist_processor is None:
            logger.warning("Listas no cargadas. Llama a load() primero.")
            return None

        if not text or not text.strip():
            return None

        matches = self._blacklist_processor.extract_keywords(text)

        if not matches:
            return None

        # Devolver el match más largo (más específico)
        return max(matches, key=len)

    def find_all_in_whitelist(self, text: str) -> List[str]:
        """
        Encuentra TODOS los términos de whitelist que aparecen en el texto.

        Args:
            text: Texto donde buscar.

        Returns:
            Lista de todos los términos encontrados (puede estar vacía).
        """
        if not self._loaded or self._whitelist_processor is None:
            return []

        if not text or not text.strip():
            return []

        return self._whitelist_processor.extract_keywords(text)

    def find_all_in_blacklist(self, text: str) -> List[str]:
        """
        Encuentra TODOS los términos de blacklist que aparecen en el texto.

        Args:
            text: Texto donde buscar.

        Returns:
            Lista de todos los términos encontrados (puede estar vacía).
        """
        if not self._loaded or self._blacklist_processor is None:
            return []

        if not text or not text.strip():
            return []

        return self._blacklist_processor.extract_keywords(text)

    def get_all_data(self) -> Dict[str, Any]:
        """
        Retorna toda la estructura de datos cargada.

        Returns:
            Dict con whitelists, blacklists y listas globales.
        """
        return {
            "whitelists": self.whitelists,
            "blacklists": self.blacklists,
            "whitelist_global": list(self.whitelist_global),
            "blacklist_global": list(self.blacklist_global)
        }

    def get_stats(self) -> Dict[str, Any]:
        """
        Retorna estadísticas de la carga.

        Returns:
            Dict con estadísticas.
        """
        return {
            **self._stats,
            "whitelist_terms_total": len(self.whitelist_global),
            "blacklist_terms_total": len(self.blacklist_global)
        }

    def __repr__(self) -> str:
        """Representación string del manager."""
        return (
            f"CsvListManager("
            f"path='{self.base_path}', "
            f"whitelist={len(self.whitelist_global)}, "
            f"blacklist={len(self.blacklist_global)})"
        )


# ============================================================================
# FUNCIÓN DE CARGA STANDALONE
# ============================================================================

def load_csv_lists(
    base_path: str = "LISTAS/",
    lowercase_whitelist: bool = False,
    lowercase_blacklist: bool = True
) -> Dict[str, Any]:
    """
    Función de conveniencia para cargar listas sin instanciar la clase.

    Args:
        base_path: Ruta a la carpeta con archivos CSV/Excel.
        lowercase_whitelist: Si True, normaliza whitelist a minúsculas.
        lowercase_blacklist: Si True, normaliza blacklist a minúsculas.

    Returns:
        Dict con estructura:
        {
            "whitelists": {nombre_archivo: [valores...]},
            "blacklists": {nombre_archivo: [valores...]},
            "whitelist_global": [todos los valores whitelist],
            "blacklist_global": [todos los valores blacklist]
        }

    Example:
        >>> data = load_csv_lists("LISTAS/")
        >>> len(data["whitelist_global"])
        1500
    """
    manager = CsvListManager(
        base_path=base_path,
        lowercase_whitelist=lowercase_whitelist,
        lowercase_blacklist=lowercase_blacklist
    )
    manager.load()
    return manager.get_all_data()


# ============================================================================
# MAIN - DEMO Y TESTS
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("CsvListManager - Demo")
    print("=" * 80)

    # Crear instancia
    manager = CsvListManager("LISTAS/")

    # Cargar listas
    print("\n[1] Cargando listas...\n")
    try:
        manager.load()
    except FileNotFoundError as e:
        print(f"  ✗ Error: {e}")
        print("  Creando carpeta LISTAS/ de ejemplo...")
        Path("LISTAS").mkdir(exist_ok=True)
        print("  Por favor, añade archivos CSV/Excel a la carpeta LISTAS/")
        exit(1)

    # Mostrar estadísticas
    print(f"\n[2] Estadísticas:\n")
    stats = manager.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Mostrar estructura
    print(f"\n[3] Listas cargadas:\n")
    print(f"  Whitelists:")
    for name, values in manager.whitelists.items():
        print(f"    - {name}: {len(values)} términos")
        if values:
            print(f"      Ejemplos: {values[:3]}")

    print(f"\n  Blacklists:")
    for name, values in manager.blacklists.items():
        print(f"    - {name}: {len(values)} términos")
        if values:
            print(f"      Ejemplos: {values[:3]}")

    # Tests de consulta
    print(f"\n[4] Tests de consulta:\n")

    # Obtener algunos términos de ejemplo de las listas cargadas
    whitelist_samples = list(manager.whitelist_global)[:5] if manager.whitelist_global else []
    blacklist_samples = list(manager.blacklist_global)[:5] if manager.blacklist_global else []

    print(f"  Whitelist samples: {whitelist_samples}")
    print(f"  Blacklist samples: {blacklist_samples}")

    # Test whitelist
    if whitelist_samples:
        test_term = whitelist_samples[0]
        result = manager.is_in_whitelist(test_term)
        print(f"\n  is_in_whitelist('{test_term}'): {result}")

    # Test blacklist
    if blacklist_samples:
        test_term = blacklist_samples[0]
        result = manager.is_in_blacklist(test_term)
        print(f"  is_in_blacklist('{test_term}'): {result}")

    # Test término inexistente
    print(f"  is_in_whitelist('INEXISTENTE_12345'): {manager.is_in_whitelist('INEXISTENTE_12345')}")
    print(f"  is_in_blacklist('INEXISTENTE_12345'): {manager.is_in_blacklist('INEXISTENTE_12345')}")

    print("\n" + "=" * 80)
    print(f"  {manager}")
    print("=" * 80)
