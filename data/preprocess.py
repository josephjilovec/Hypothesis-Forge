import os
import pandas as pd
import numpy as np
from astropy.io import fits
from Bio.PDB import PDBParser, PPBuilder
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataPreprocessor:
    """Handles ingestion, cleaning, and formatting of scientific datasets."""
    
    def __init__(self, raw_dir='data/raw', processed_dir='data/processed'):
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
    def process_pdb_file(self, pdb_path):
        """Process AlphaFold PDB files to extract sequence and structural data."""
        try:
            parser = PDBParser(QUIET=True)
            structure = parser.get_structure('protein', pdb_path)
            ppb = PPBuilder()
            
            # Extract sequence
            sequences = [pp.get_sequence() for pp in ppb.build_peptides(structure)]
            sequence_data = [str(seq) for seq in sequences]
            
            # Extract coordinates (example: CA atoms)
            coords = []
            for model in structure:
                for chain in model:
                    for residue in chain:
                        if 'CA' in residue:
                            coords.append(residue['CA'].get_coord())
            
            # Convert to DataFrame
            df = pd.DataFrame({
                'sequence': sequence_data,
                'coords': [coords[:len(sequence_data)]] * len(sequence_data)
            })
            
            # Handle missing values
            df['sequence'] = df['sequence'].replace('', np.nan)
            df = df.dropna()
            
            return df
        
        except Exception as e:
            logger.error(f"Error processing PDB file {pdb_path}: {str(e)}")
            return None
    
    def process_fits_file(self, fits_path):
        """Process NASA FITS files to extract astronomical data."""
        try:
            with fits.open(fits_path) as hdul:
                # Example: Extract data from first HDU with data
                for hdu in hdul:
                    if hdu.data is not None:
                        data = hdu.data
                        break
                else:
                    raise ValueError("No data found in FITS file")
                
                # Convert to DataFrame
                if data.ndim == 1:
                    df = pd.DataFrame(data)
                else:
                    df = pd.DataFrame(data.flatten().reshape(-1, data.shape[-1]))
                
                # Handle missing values and normalize
                df = df.fillna(df.mean())
                df = (df - df.mean()) / df.std()
                
                return df
        
        except Exception as e:
            logger.error(f"Error processing FITS file {fits_path}: {str(e)}")
            return None
    
    def process_dataset(self, file_path):
        """Dispatch processing based on file extension."""
        file_ext = file_path.suffix.lower()
        
        if file_ext == '.pdb':
            return self.process_pdb_file(file_path)
        elif file_ext == '.fits':
            return self.process_fits_file(file_path)
        else:
            logger.error(f"Unsupported file format: {file_ext}")
            return None
    
    def save_processed_data(self, df, output_path):
        """Save processed DataFrame to parquet format."""
        try:
            df.to_parquet(output_path, index=False)
            logger.info(f"Saved processed data to {output_path}")
        except Exception as e:
            logger.error(f"Error saving processed data to {output_path}: {str(e)}")
    
    def run(self):
        """Process all datasets in raw_dir and save to processed_dir."""
        for subdir in ['alphafold', 'nasa']:
            subdir_path = self.raw_dir / subdir
            if not subdir_path.exists():
                logger.warning(f"Directory {subdir_path} does not exist")
                continue
                
            for file_path in subdir_path.glob('*'):
                logger.info(f"Processing {file_path}")
                df = self.process_dataset(file_path)
                
                if df is not None and not df.empty:
                    output_path = self.processed_dir / f"{file_path.stem}_processed.parquet"
                    self.save_processed_data(df, output_path)
                else:
                    logger.warning(f"No valid data processed for {file_path}")

if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    preprocessor.run()
