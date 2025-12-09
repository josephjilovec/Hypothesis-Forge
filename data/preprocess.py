"""
Data preprocessing module for Hypothesis Forge.
Handles PDB (protein) and FITS (astrophysical) file processing.
"""
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from Bio.PDB import PDBParser, MMCIFParser
from Bio.PDB.PDBExceptions import PDBConstructionWarning
import warnings
from astropy.io import fits
from astropy.table import Table

from config.config import RAW_DATA_DIR, PROCESSED_DATA_DIR
from utils.logging_config import logger
from utils.error_handling import retry, handle_errors

# Suppress BioPython warnings
warnings.simplefilter("ignore", PDBConstructionWarning)


class ProteinPreprocessor:
    """Preprocesses PDB protein structure files."""

    def __init__(self, pdb_dir: Path):
        """
        Initialize protein preprocessor.

        Args:
            pdb_dir: Directory containing PDB files
        """
        self.pdb_dir = Path(pdb_dir)
        self.parser = PDBParser(QUIET=True)
        self.mmcif_parser = MMCIFParser(QUIET=True)

    @retry(max_attempts=2, delay=1.0)
    def extract_protein_features(self, pdb_file: Path) -> Optional[Dict[str, Any]]:
        """
        Extract features from a PDB file.

        Args:
            pdb_file: Path to PDB file

        Returns:
            Dictionary of extracted features or None if parsing fails
        """
        try:
            structure_id = pdb_file.stem
            structure = self.parser.get_structure(structure_id, str(pdb_file))

            features = {
                "structure_id": structure_id,
                "num_chains": len(list(structure.get_chains())),
                "num_residues": len(list(structure.get_residues())),
                "num_atoms": len(list(structure.get_atoms())),
            }

            # Extract chain information
            chains_info = []
            for chain in structure.get_chains():
                chain_id = chain.get_id()
                residues = list(chain.get_residues())
                atoms = list(chain.get_atoms())

                chains_info.append({
                    "chain_id": chain_id,
                    "num_residues": len(residues),
                    "num_atoms": len(atoms),
                })

            features["chains"] = chains_info

            # Calculate geometric features
            coords = []
            for atom in structure.get_atoms():
                coords.append(atom.get_coord())

            if coords:
                coords = np.array(coords)
                features["center_of_mass"] = coords.mean(axis=0).tolist()
                features["radius_of_gyration"] = float(
                    np.sqrt(np.mean(np.sum((coords - coords.mean(axis=0)) ** 2, axis=1)))
                )
                features["max_dimension"] = float(np.max(coords.max(axis=0) - coords.min(axis=0)))

            return features

        except Exception as e:
            logger.warning(f"Failed to parse {pdb_file}: {e}")
            return None

    def process_all(self) -> pd.DataFrame:
        """
        Process all PDB files in the directory.

        Returns:
            DataFrame with extracted features
        """
        pdb_files = list(self.pdb_dir.glob("*.pdb")) + list(self.pdb_dir.glob("*.ent"))
        cif_files = list(self.pdb_dir.glob("*.cif"))

        all_features = []

        for pdb_file in pdb_files:
            features = self.extract_protein_features(pdb_file)
            if features:
                all_features.append(features)

        for cif_file in cif_files:
            try:
                structure_id = cif_file.stem
                structure = self.mmcif_parser.get_structure(structure_id, str(cif_file))
                features = {
                    "structure_id": structure_id,
                    "num_chains": len(list(structure.get_chains())),
                    "num_residues": len(list(structure.get_residues())),
                    "num_atoms": len(list(structure.get_atoms())),
                }
                all_features.append(features)
            except Exception as e:
                logger.warning(f"Failed to parse {cif_file}: {e}")

        if not all_features:
            logger.warning("No valid protein structures found")
            return pd.DataFrame()

        df = pd.DataFrame(all_features)
        logger.info(f"Processed {len(df)} protein structures")
        return df


class FITSPreprocessor:
    """Preprocesses FITS astrophysical data files."""

    def __init__(self, fits_dir: Path):
        """
        Initialize FITS preprocessor.

        Args:
            fits_dir: Directory containing FITS files
        """
        self.fits_dir = Path(fits_dir)

    def extract_fits_features(self, fits_file: Path) -> Optional[Dict[str, Any]]:
        """
        Extract features from a FITS file.

        Args:
            fits_file: Path to FITS file

        Returns:
            Dictionary of extracted features or None if parsing fails
        """
        try:
            with fits.open(str(fits_file)) as hdul:
                features = {
                    "file_name": fits_file.name,
                    "num_hdus": len(hdul),
                }

                # Extract header information
                primary_header = hdul[0].header
                features["header_keys"] = len(primary_header)

                # Extract data shape if available
                if hdul[0].data is not None:
                    data = hdul[0].data
                    features["data_shape"] = list(data.shape) if hasattr(data, "shape") else None
                    features["data_dtype"] = str(data.dtype) if hasattr(data, "dtype") else None

                    if isinstance(data, np.ndarray):
                        features["data_min"] = float(np.nanmin(data))
                        features["data_max"] = float(np.nanmax(data))
                        features["data_mean"] = float(np.nanmean(data))
                        features["data_std"] = float(np.nanstd(data))

                # Extract table data if present
                if len(hdul) > 1:
                    tables = []
                    for i, hdu in enumerate(hdul[1:], 1):
                        if isinstance(hdu, fits.BinTableHDU):
                            table = Table.read(hdu)
                            tables.append({
                                "hdu_index": i,
                                "num_rows": len(table),
                                "num_cols": len(table.columns),
                                "column_names": list(table.columns.names),
                            })
                    features["tables"] = tables

                return features

        except Exception as e:
            logger.warning(f"Failed to parse {fits_file}: {e}")
            return None

    def process_all(self) -> pd.DataFrame:
        """
        Process all FITS files in the directory.

        Returns:
            DataFrame with extracted features
        """
        fits_files = list(self.fits_dir.glob("*.fits")) + list(self.fits_dir.glob("*.fit"))

        all_features = []

        for fits_file in fits_files:
            features = self.extract_fits_features(fits_file)
            if features:
                all_features.append(features)

        if not all_features:
            logger.warning("No valid FITS files found")
            return pd.DataFrame()

        df = pd.DataFrame(all_features)
        logger.info(f"Processed {len(df)} FITS files")
        return df


def preprocess_data():
    """
    Main preprocessing function.
    Processes both protein and astrophysical data.
    """
    logger.info("Starting data preprocessing...")

    # Process protein data
    alphafold_dir = RAW_DATA_DIR / "alphafold"
    if alphafold_dir.exists() and any(alphafold_dir.iterdir()):
        logger.info("Processing AlphaFold protein structures...")
        protein_pp = ProteinPreprocessor(alphafold_dir)
        protein_df = protein_pp.process_all()

        if not protein_df.empty:
            output_path = PROCESSED_DATA_DIR / "alphafold_processed.parquet"
            protein_df.to_parquet(output_path, index=False)
            logger.info(f"Saved processed protein data to {output_path}")
    else:
        logger.info("No AlphaFold data found, creating sample structure...")
        # Create sample data structure
        sample_protein_df = pd.DataFrame([{
            "structure_id": "sample_protein",
            "num_chains": 1,
            "num_residues": 100,
            "num_atoms": 500,
            "chains": [{"chain_id": "A", "num_residues": 100, "num_atoms": 500}],
            "center_of_mass": [0.0, 0.0, 0.0],
            "radius_of_gyration": 10.0,
            "max_dimension": 20.0,
        }])
        output_path = PROCESSED_DATA_DIR / "alphafold_processed.parquet"
        sample_protein_df.to_parquet(output_path, index=False)
        logger.info(f"Created sample protein data at {output_path}")

    # Process astrophysical data
    nasa_dir = RAW_DATA_DIR / "nasa"
    if nasa_dir.exists() and any(nasa_dir.iterdir()):
        logger.info("Processing NASA FITS files...")
        fits_pp = FITSPreprocessor(nasa_dir)
        fits_df = fits_pp.process_all()

        if not fits_df.empty:
            output_path = PROCESSED_DATA_DIR / "nasa_processed.parquet"
            fits_df.to_parquet(output_path, index=False)
            logger.info(f"Saved processed FITS data to {output_path}")
    else:
        logger.info("No NASA FITS data found, creating sample structure...")
        # Create sample data structure
        sample_fits_df = pd.DataFrame([{
            "file_name": "sample_observation.fits",
            "num_hdus": 2,
            "header_keys": 50,
            "data_shape": [1024, 1024],
            "data_dtype": "float32",
            "data_min": 0.0,
            "data_max": 1000.0,
            "data_mean": 500.0,
            "data_std": 200.0,
            "tables": [{"hdu_index": 1, "num_rows": 100, "num_cols": 5, "column_names": ["ra", "dec", "flux", "error", "snr"]}],
        }])
        output_path = PROCESSED_DATA_DIR / "nasa_processed.parquet"
        sample_fits_df.to_parquet(output_path, index=False)
        logger.info(f"Created sample FITS data at {output_path}")

    logger.info("Data preprocessing complete!")


if __name__ == "__main__":
    preprocess_data()

