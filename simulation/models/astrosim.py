import torch
import torch.nn as nn
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GravitationalModel(nn.Module):
    """Neural network for approximating gravitational interactions."""
    def __init__(self, input_dim=4, hidden_dim=128, output_dim=3):
        super(GravitationalModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.network(x)

class SimulationModel:
    """Astrophysical simulation for gravitational dynamics using NASA FITS data."""
    
    def __init__(self, device='cpu'):
        self.device = torch.device(device)
        self.model = GravitationalModel().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.G = 6.67430e-11  # Gravitational constant (m^3 kg^-1 s^-2)
        logger.info(f"Initialized GravitationalModel on {self.device}")
    
    def calculate_nbody_forces(self, positions, masses):
        """Calculate gravitational forces using N-body physics."""
        try:
            n = positions.shape[0]
            forces = torch.zeros_like(positions, device=self.device)
            
            for i in range(n):
                for j in range(n):
                    if i != j:
                        # Vector from body i to j
                        r_vec = positions[j] - positions[i]
                        r = torch.norm(r_vec, dim=-1, keepdim=True)
                        r = torch.clamp(r, min=1e-6)  # Avoid division by zero
                        
                        # Gravitational force: F = G * m1 * m2 / r^2
                        force_magnitude = self.G * masses[i] * masses[j] / (r ** 2)
                        force_direction = r_vec / r
                        forces[i] += force_magnitude * force_direction
            
            return forces
        except Exception as e:
            logger.error(f"Error calculating N-body forces: {str(e)}")
            return None
    
    def simulate(self, data):
        """Run astrophysical simulation on input data (positions, masses)."""
        try:
            # Assume data is a tensor of shape (n_bodies, 4) [x, y, z, mass]
            if data.ndim != 2 or data.shape[1] < 4:
                raise ValueError(f"Expected shape (n, >=4), got {data.shape}")
            
            positions = data[:, :3].to(self.device)  # x, y, z coordinates
            masses = data[:, 3].to(self.device)  # Mass of each body
            velocities = torch.zeros_like(positions, device=self.device)
            
            # Simulation parameters
            dt = 1e3  # Time step in seconds (optimized for <5 min runtime)
            max_steps = 500  # Optimized for performance
            mass_scale = 1e30  # Typical stellar mass (kg, e.g., solar masses)
            
            # Initialize positions and velocities
            positions = positions.clone().detach().requires_grad_(True)
            masses = masses.clone().detach() * mass_scale
            
            for step in range(max_steps):
                self.optimizer.zero_grad()
                
                # Calculate physical forces
                forces = self.calculate_nbody_forces(positions, masses)
                if forces is None:
                    return None
                
                # Predict velocity adjustments using neural network
                input_data = torch.cat([positions, masses.unsqueeze(-1)], dim=-1)
                velocity_adjustments = self.model(input_data)
                velocities = velocities + velocity_adjustments
                
                # Update positions: r = r + v * dt
                new_positions = positions + velocities * dt
                accelerations = forces / masses.unsqueeze(-1)
                velocities = velocities + accelerations * dt
                
                # Compute loss (e.g., minimize energy drift)
                energy = self.calculate_energy(positions, velocities, masses)
                if energy is None:
                    return None
                
                energy.backward()
                self.optimizer.step()
                
                # Update positions
                positions = new_positions.detach().requires_grad_(True)
                
                # Log progress
                if step % 100 == 0:
                    logger.info(f"Step {step}, Energy: {energy.item():.4f}")
                
                # Early stopping if energy converges
                if step > 0 and torch.abs(energy - prev_energy) < 1e-4:
                    logger.info(f"Converged at step {step}")
                    break
                prev_energy = energy
            
            # Return updated positions and velocities
            return torch.cat([positions, velocities], dim=-1).detach().cpu()
        
        except Exception as e:
            logger.error(f"Simulation failed: {str(e)}")
            return None
    
    def calculate_energy(self, positions, velocities, masses):
        """Calculate total energy (kinetic + potential) of the system."""
        try:
            # Kinetic energy: K = 0.5 * m * v^2
            kinetic = 0.5 * torch.sum(masses * torch.sum(velocities ** 2, dim=-1))
            
            # Potential energy: U = -G * m1 * m2 / r
            potential = 0.0
            n = positions.shape[0]
            for i in range(n):
                for j in range(i + 1, n):
                    r_vec = positions[j] - positions[i]
                    r = torch.norm(r_vec, dim=-1)
                    r = torch.clamp(r, min=1e-6)
                    potential -= self.G * masses[i] * masses[j] / r
            
            return kinetic + potential
        except Exception as e:
            logger.error(f"Error calculating energy: {str(e)}")
            return None

if __name__ == "__main__":
    # Example usage for testing
    sim = SimulationModel(device='cuda' if torch.cuda.is_available() else 'cpu')
    # Dummy input: 5 bodies with [x, y, z, mass]
    test_data = torch.randn(5, 4)
    results = sim.simulate(test_data)
    if results is not None:
        logger.info(f"Simulation results shape: {results.shape}")
