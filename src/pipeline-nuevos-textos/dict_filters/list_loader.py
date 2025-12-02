#!/usr/bin/env python3
"""
ListLoader - Gestor centralizado de listas blancas/negras.

Carga y gestiona listas de filtrado desde múltiples fuentes:
- Archivos JSON (hospitales, lugares, medicamentos, patologías)
- Archivos CSV/Excel
- Archivo CIE10 especial

Proporciona una API simple y consistente para consultas.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Any

logger = logging.getLogger(__name__)


# Columnas conocidas para archivos CSV
KNOWN_COLUMN_MAPPINGS: Dict[str, List[str]] = {
    "CNH": ["Nombre Centro", "Municipio", "Provincia", "CCAA", "Nombre del Complejo"],
    "nomenclator": ["DENOMINACIÓN", "denominacion", "Denominación", "Nombre", "nombre"],
    "codmun": ["NOMBRE", "Nombre", "MUNICIPIO", "Municipio"],
}

GENERIC_COLUMNS = [
    "nombre", "name", "denominacion", "denominación",
    "texto", "text", "termino", "term", "entidad", "entity"
]


class ListLoader:
    """
    Gestor centralizado de listas blancas y negras para filtrado de entidades.
    
    Carga automáticamente desde múltiples fuentes y proporciona API unificada.
    """
    
    def __init__(
        self,
        json_whitelist_paths: Optional[List[str]] = None,
        json_blacklist_paths: Optional[List[str]] = None,
        csv_base_path: Optional[str] = None,
        cie10_path: Optional[str] = None,
        lowercase_whitelist: bool = False,
        lowercase_blacklist: bool = True
    ):
        """
        Inicializa el gestor de listas.
        
        Args:
            json_whitelist_paths: Rutas a archivos JSON de whitelist
            json_blacklist_paths: Rutas a archivos JSON de blacklist
            csv_base_path: Ruta a carpeta con archivos CSV/Excel
            cie10_path: Ruta al archivo CIE10
            lowercase_whitelist: Si True, normaliza whitelist a minúsculas
            lowercase_blacklist: Si True, normaliza blacklist a minúsculas
        """
        self.lowercase_whitelist = lowercase_whitelist
        self.lowercase_blacklist = lowercase_blacklist
        
        # Sets principales
        self.whitelist_set: Set[str] = set()
        self.blacklist_set: Set[str] = set()
        self.cie10_set: Set[str] = set()
        
        # Diccionarios por fuente (para debugging)
        self.whitelist_by_source: Dict[str, Set[str]] = {}
        self.blacklist_by_source: Dict[str, Set[str]] = {}
        
        # Procesadores flashtext
        self._whitelist_processor = None
        self._blacklist_processor = None
        
        # Cargar todas las fuentes
        self._load_all_sources(
            json_whitelist_paths,
            json_blacklist_paths,
            csv_base_path,
            cie10_path
        )
        
        # Inicializar procesadores flashtext
        self._init_flashtext_processors()
        
        logger.info(
            f"ListLoader initialized: "
            f"whitelist={len(self.whitelist_set)}, "
            f"blacklist={len(self.blacklist_set)}, "
            f"cie10={len(self.cie10_set)}"
        )
    
    def _load_all_sources(
        self,
        json_whitelist_paths: Optional[List[str]],
        json_blacklist_paths: Optional[List[str]],
        csv_base_path: Optional[str],
        cie10_path: Optional[str]
    ):
        """Carga todas las fuentes de listas."""
        # JSON whitelists
        if json_whitelist_paths:
            for path in json_whitelist_paths:
                if Path(path).exists():
                    terms = self._load_json_file(path, self.lowercase_whitelist)
                    self.whitelist_set.update(terms)
                    self.whitelist_by_source[Path(path).name] = terms
                    logger.info(f"Loaded {len(terms)} whitelist terms from {Path(path).name}")
        
        # JSON blacklists
        if json_blacklist_paths:
            for path in json_blacklist_paths:
                if Path(path).exists():
                    terms = self._load_json_file(path, self.lowercase_blacklist)
                    self.blacklist_set.update(terms)
                    self.blacklist_by_source[Path(path).name] = terms
                    logger.info(f"Loaded {len(terms)} blacklist terms from {Path(path).name}")
        
        # CSV/Excel
        if csv_base_path and Path(csv_base_path).exists():
            self._load_csv_excel_folder(csv_base_path)
        
        # CIE10
        if cie10_path and Path(cie10_path).exists():
            self.cie10_set = self._load_cie10_file(cie10_path)
            logger.info(f"Loaded {len(self.cie10_set)} CIE10 terms")
    
    def _load_json_file(self, path: str, lowercase: bool = False) -> Set[str]:
        """Carga términos desde un archivo JSON."""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            terms = set()
            self._extract_strings_recursive(data, terms)
            
            if lowercase:
                terms = {t.lower() for t in terms}
            
            return terms
            
        except Exception as e:
            logger.error(f"Error loading JSON {path}: {e}")
            return set()
    
    def _extract_strings_recursive(self, obj: Any, accumulator: Set[str]):
        """Extrae strings de una estructura JSON anidada."""
        if isinstance(obj, str):
            if obj.strip():
                accumulator.add(obj.strip())
        elif isinstance(obj, list):
            for item in obj:
                self._extract_strings_recursive(item, accumulator)
        elif isinstance(obj, dict):
            for value in obj.values():
                self._extract_strings_recursive(value, accumulator)
    
    def _load_csv_excel_folder(self, folder_path: str):
        """Carga todos los archivos CSV/Excel de una carpeta."""
        try:
            import pandas as pd
        except ImportError:
            logger.warning("pandas not installed, skipping CSV/Excel loading")
            return
        
        folder = Path(folder_path)
        extensions = ['*.csv', '*.xls', '*.xlsx']
        
        for ext in extensions:
            for file_path in folder.glob(ext):
                self._process_csv_excel_file(file_path)
    
    def _process_csv_excel_file(self, file_path: Path):
        """Procesa un archivo CSV/Excel individual."""
        try:
            import pandas as pd
            
            # Leer archivo
            if file_path.suffix == '.csv':
                df = pd.read_csv(file_path, encoding='utf-8')
            else:
                df = pd.read_excel(file_path)
            
            if df.empty:
                return
            
            # Determinar tipo
            file_name_lower = file_path.name.lower()
            is_whitelist = any(
                keyword in file_name_lower
                for keyword in ['hospital', 'lugar', 'cnh', 'nomenclator', 'municipio', 'codmun']
            )
            is_blacklist = any(
                keyword in file_name_lower
                for keyword in ['medicamento', 'farmaco', 'fármaco', 'patolog']
            )
            
            if not is_whitelist and not is_blacklist:
                logger.warning(f"Cannot classify {file_path.name}, skipping")
                return
            
            # Detectar columnas
            columns = self._detect_relevant_columns(file_path.name, df)
            if not columns:
                return
            
            # Extraer valores
            lowercase = self.lowercase_blacklist if is_blacklist else self.lowercase_whitelist
            terms = self._extract_values(df, columns, lowercase)
            
            # Añadir
            if is_whitelist:
                self.whitelist_set.update(terms)
                self.whitelist_by_source[file_path.name] = terms
            else:
                self.blacklist_set.update(terms)
                self.blacklist_by_source[file_path.name] = terms
            
            logger.info(f"Loaded {len(terms)} terms from {file_path.name}")
                
        except Exception as e:
            logger.error(f"Error processing {file_path.name}: {e}")
    
    def _detect_relevant_columns(self, file_name: str, df) -> List[str]:
        """Detecta las columnas relevantes de un DataFrame."""
        # Mapeo conocido
        for pattern, columns in KNOWN_COLUMN_MAPPINGS.items():
            if pattern.lower() in file_name.lower():
                matching_cols = [col for col in columns if col in df.columns]
                if matching_cols:
                    return matching_cols
        
        # Columnas genéricas
        generic_cols = []
        for col in df.columns:
            if any(generic in str(col).lower() for generic in GENERIC_COLUMNS):
                generic_cols.append(col)
        
        if generic_cols:
            return generic_cols
        
        # Fallback
        for col in df.columns:
            if not df[col].isna().all():
                return [col]
        
        return []
    
    def _extract_values(self, df, columns: List[str], lowercase: bool = False) -> Set[str]:
        """Extrae y normaliza valores de las columnas."""
        values = set()
        
        for col in columns:
            if col not in df.columns:
                continue
            
            col_values = df[col].dropna().astype(str)
            
            for val in col_values:
                val = val.strip()
                if not val or val.isdigit():
                    continue
                
                if lowercase:
                    val = val.lower()
                
                values.add(val)
        
        return values
    
    def _load_cie10_file(self, cie10_path: str) -> Set[str]:
        """Carga términos médicos desde archivo CIE10."""
        try:
            import pandas as pd
            
            df = pd.read_excel(cie10_path)
            
            desc_columns = [
                col for col in df.columns
                if any(keyword in str(col).lower() 
                       for keyword in ['descripcion', 'description', 'nombre', 'name'])
            ]
            
            if not desc_columns:
                desc_columns = [col for col in df.columns if df[col].dtype == 'object']
            
            terms = set()
            for col in desc_columns:
                col_values = df[col].dropna().astype(str)
                for val in col_values:
                    val = val.strip().lower()
                    if val and len(val) > 2 and not val.isdigit():
                        terms.add(val)
            
            return terms
            
        except Exception as e:
            logger.error(f"Error loading CIE10 {cie10_path}: {e}")
            return set()
    
    def _init_flashtext_processors(self):
        """Inicializa los procesadores flashtext."""
        try:
            from flashtext import KeywordProcessor
            
            self._whitelist_processor = KeywordProcessor(case_sensitive=not self.lowercase_whitelist)
            for term in self.whitelist_set:
                self._whitelist_processor.add_keyword(term)
            
            self._blacklist_processor = KeywordProcessor(case_sensitive=not self.lowercase_blacklist)
            for term in self.blacklist_set:
                self._blacklist_processor.add_keyword(term)
            
            logger.debug("Flashtext processors initialized")
        except ImportError:
            logger.warning("flashtext not installed, substring search will be slower")
    
    # ========================================================================
    # API PÚBLICA
    # ========================================================================
    
    def is_in_whitelist(self, text: str) -> bool:
        """Verifica si un texto está EXACTAMENTE en la whitelist."""
        search_text = text.lower() if self.lowercase_whitelist else text
        return search_text in self.whitelist_set
    
    def is_in_blacklist(self, text: str) -> bool:
        """Verifica si un texto está EXACTAMENTE en la blacklist."""
        search_text = text.lower() if self.lowercase_blacklist else text
        return search_text in self.blacklist_set
    
    def is_in_cie10(self, text: str) -> bool:
        """Verifica si un texto está en el CIE10."""
        return text.lower() in self.cie10_set
    
    def find_in_whitelist(self, text: str) -> Optional[str]:
        """Busca si algún término de la whitelist aparece DENTRO del texto."""
        if self._whitelist_processor:
            keywords_found = self._whitelist_processor.extract_keywords(text)
            return keywords_found[0] if keywords_found else None
        return None
    
    def find_in_blacklist(self, text: str) -> Optional[str]:
        """Busca si algún término de la blacklist aparece DENTRO del texto."""
        if self._blacklist_processor:
            keywords_found = self._blacklist_processor.extract_keywords(text)
            return keywords_found[0] if keywords_found else None
        return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Retorna estadísticas del gestor de listas."""
        return {
            "whitelist_terms": len(self.whitelist_set),
            "blacklist_terms": len(self.blacklist_set),
            "cie10_terms": len(self.cie10_set),
            "cie10_loaded": len(self.cie10_set) > 0,
            "whitelist_sources": list(self.whitelist_by_source.keys()),
            "blacklist_sources": list(self.blacklist_by_source.keys()),
        }
