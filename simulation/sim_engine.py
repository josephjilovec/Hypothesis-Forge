import os
import time
import pandas as pd
import torch
import logging
from pathlib import Path
from importlib import import_module
import sys

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimulationEngine:
    """Orchestrates simulations for scientific systems using preprocessed data."""
    
    def __init__(self, data_dir='data/processed', model_dir='simulation/models', output_dir='simulation/results'):
        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
    def load_data(self, data_path):
        """Load preprocessed data from parquet file."""
        try:
            df = pd.read_parquet(data_path)
            if df.empty:
                raise ValueError("Loaded DataFrame is empty")
            return df
        except Exception as e:
            logger.error(f"Error loading data from {data_path}: {str(e)}")
            return None
    
    def load_model(self, model_name):
        """Dynamically load simulation model from models directory."""
        try:
            sys.path.append(str(self.model_dir))
            model_module = import_module(model_name)
            model = getattr(model_module, 'SimulationModel')(device=self.device)
            return model
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {str(e)}")
            return None
    
    def run_simulation(self, data, model, max_duration=300):
        """Run simulation with timeout to ensure <5 min execution."""
        start_time = time.time()
        try:
            # Convert data to tensor if needed
            if isinstance(data, pd.DataFrame):
                data_tensor = torch.tensor(data.values, dtype=torch.float32).to(self.device)
            else:
                data_tensor = data.to(self.device)
            
            # Run model
            with torch.no_grad():
                results = model.simulate(data_tensor)
            
            # Check timeout
            elapsed = time.time() - start_time
            if elapsed > max_duration:
                raise TimeoutError(f"Simulation exceeded {max_duration}s")
                
            return results.cpu().numpy()
        except Exception as e:
            logger.error(f"Simulation failed: {str(e)}")
            return None
    
    def save_results(self, results, output_path):
        """Save simulation results to parquet file."""
        try:
            if results is not None:
                df = pd.DataFrame(results)
                df.to_parquet(output_path, index=False)
                logger.info(f"Saved results to {output_path}")
            else:
                logger.warning(f"No results to save for {output_path}")
        except Exception as e:
            logger.error(f"Error saving results to {output_path}: {str(e)}")
    
    def run(self):
        """Orchestrate simulations for all preprocessed datasets."""
        model_mapping = {
            'protein': 'proteinsim',
            'astro': 'astrosim'
        }
        
        for data_path in self.data_dir.glob('*.parquet'):
            # Determine simulation type from filename
            sim_type = 'protein' if 'protein' in data_path.stem.lower() else 'astro'
            model_name = model_mapping.get(sim_type)
            
            if not model_name:
                logger.warning(f"No model mapped for {data_path}")
                continue
                
            logger.info(f"Processing {data_path} with {model_name}")
            
            # Load data
            data = self.load_data(data_path)
            if data is None:
                continue
                
            # Load model
            model = self.load_model(model_name)
            if model is None:
                continue
                
            # Run simulation
            results = self.run_simulation(data, model)
            if results is None:
                continue
                
            # Save results
            output_path = self.output_dir / f"{data_path.stem}_results.parquet"
            self.save_results(results, output_path)

# Example simulation model interface (to be implemented in models/*.py)
class SimulationModel:
    def __init__(self, device):
        self.device = device
    
    def simulate(self, data):
        """Placeholder for simulation logic."""
        raise NotImplementedError("Simulation model must implement simulate method")

if __name__ == "__main__":
    engine = SimulationEngine()
    engine.run()
