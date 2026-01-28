"""
1D Ising Model Simulation

This module implements a Monte Carlo simulation of the one-dimensional Ising model
using the Metropolis algorithm. The Ising model is a mathematical model of
ferromagnetism in statistical mechanics.

The Hamiltonian of the 1D Ising model is:
    H = -J * Σ(s_i * s_{i+1}) - h * Σ(s_i)

where:
    - J is the coupling constant (interaction strength between neighbors)
    - h is the external magnetic field
    - s_i is the spin at site i (+1 or -1)

Author: AI Examples Repository
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional
from dataclasses import dataclass


@dataclass
class SimulationResults:
    """Container for simulation results."""
    temperatures: np.ndarray
    energies: np.ndarray
    magnetizations: np.ndarray
    specific_heats: np.ndarray
    susceptibilities: np.ndarray
    energy_stds: np.ndarray
    magnetization_stds: np.ndarray


class IsingModel1D:
    """
    1D Ising Model simulation using the Metropolis Monte Carlo algorithm.

    Attributes:
        N (int): Number of spins in the chain
        J (float): Coupling constant (J > 0 for ferromagnetic)
        h (float): External magnetic field
        spins (np.ndarray): Array of spins (+1 or -1)
    """

    def __init__(self, N: int = 100, J: float = 1.0, h: float = 0.0):
        """
        Initialize the 1D Ising model.

        Args:
            N: Number of spins in the chain
            J: Coupling constant (default: 1.0, ferromagnetic)
            h: External magnetic field (default: 0.0)
        """
        self.N = N
        self.J = J
        self.h = h
        self.spins = np.random.choice([-1, 1], size=N)

    def initialize_spins(self, state: str = "random") -> None:
        """
        Initialize the spin configuration.

        Args:
            state: "random" for random configuration,
                   "up" for all spins up (+1),
                   "down" for all spins down (-1)
        """
        if state == "random":
            self.spins = np.random.choice([-1, 1], size=self.N)
        elif state == "up":
            self.spins = np.ones(self.N, dtype=int)
        elif state == "down":
            self.spins = -np.ones(self.N, dtype=int)
        else:
            raise ValueError(f"Unknown state: {state}. Use 'random', 'up', or 'down'.")

    def calculate_energy(self) -> float:
        """
        Calculate the total energy of the current spin configuration.

        Uses periodic boundary conditions (PBC).

        Returns:
            Total energy of the system
        """
        # Interaction energy with nearest neighbors (PBC)
        interaction_energy = -self.J * np.sum(self.spins * np.roll(self.spins, 1))
        # Energy from external field
        field_energy = -self.h * np.sum(self.spins)
        return interaction_energy + field_energy

    def calculate_magnetization(self) -> float:
        """
        Calculate the total magnetization of the system.

        Returns:
            Total magnetization (sum of all spins)
        """
        return np.sum(self.spins)

    def calculate_site_energy(self, i: int) -> float:
        """
        Calculate the energy contribution of a single spin.

        Args:
            i: Index of the spin site

        Returns:
            Energy contribution of spin at site i
        """
        # Neighbors with periodic boundary conditions
        left = (i - 1) % self.N
        right = (i + 1) % self.N

        neighbor_sum = self.spins[left] + self.spins[right]
        return -self.J * self.spins[i] * neighbor_sum - self.h * self.spins[i]

    def metropolis_step(self, beta: float) -> bool:
        """
        Perform a single Metropolis Monte Carlo step.

        Args:
            beta: Inverse temperature (1 / k_B T)

        Returns:
            True if the spin flip was accepted, False otherwise
        """
        # Select a random spin
        i = np.random.randint(0, self.N)

        # Calculate energy change if we flip this spin
        left = (i - 1) % self.N
        right = (i + 1) % self.N

        neighbor_sum = self.spins[left] + self.spins[right]

        # ΔE = 2 * J * s_i * (s_{i-1} + s_{i+1}) + 2 * h * s_i
        delta_E = 2 * self.J * self.spins[i] * neighbor_sum + 2 * self.h * self.spins[i]

        # Metropolis acceptance criterion
        if delta_E <= 0 or np.random.random() < np.exp(-beta * delta_E):
            self.spins[i] *= -1
            return True
        return False

    def monte_carlo_sweep(self, beta: float) -> int:
        """
        Perform N Metropolis steps (one Monte Carlo sweep).

        Args:
            beta: Inverse temperature (1 / k_B T)

        Returns:
            Number of accepted spin flips
        """
        accepted = 0
        for _ in range(self.N):
            if self.metropolis_step(beta):
                accepted += 1
        return accepted

    def simulate(
        self,
        temperature: float,
        n_equilibrate: int = 1000,
        n_measure: int = 5000,
        measure_interval: int = 10
    ) -> Tuple[List[float], List[float]]:
        """
        Run a Monte Carlo simulation at a given temperature.

        Args:
            temperature: Temperature in units where k_B = 1
            n_equilibrate: Number of sweeps for equilibration
            n_measure: Number of sweeps for measurements
            measure_interval: Interval between measurements

        Returns:
            Tuple of (energies, magnetizations) lists
        """
        if temperature <= 0:
            raise ValueError("Temperature must be positive")

        beta = 1.0 / temperature

        # Equilibration phase
        for _ in range(n_equilibrate):
            self.monte_carlo_sweep(beta)

        # Measurement phase
        energies = []
        magnetizations = []

        for sweep in range(n_measure):
            self.monte_carlo_sweep(beta)

            if sweep % measure_interval == 0:
                energies.append(self.calculate_energy())
                magnetizations.append(self.calculate_magnetization())

        return energies, magnetizations

    def temperature_sweep(
        self,
        T_min: float = 0.5,
        T_max: float = 5.0,
        n_temps: int = 50,
        n_equilibrate: int = 1000,
        n_measure: int = 5000,
        measure_interval: int = 10,
        verbose: bool = True
    ) -> SimulationResults:
        """
        Perform simulations across a range of temperatures.

        Args:
            T_min: Minimum temperature
            T_max: Maximum temperature
            n_temps: Number of temperature points
            n_equilibrate: Equilibration sweeps per temperature
            n_measure: Measurement sweeps per temperature
            measure_interval: Interval between measurements
            verbose: Print progress if True

        Returns:
            SimulationResults object with all computed quantities
        """
        temperatures = np.linspace(T_min, T_max, n_temps)

        avg_energies = []
        avg_magnetizations = []
        specific_heats = []
        susceptibilities = []
        energy_stds = []
        magnetization_stds = []

        for i, T in enumerate(temperatures):
            if verbose and (i + 1) % 10 == 0:
                print(f"Progress: {i + 1}/{n_temps} temperatures completed")

            # Reset to random state
            self.initialize_spins("random")

            # Run simulation
            energies, magnetizations = self.simulate(
                T, n_equilibrate, n_measure, measure_interval
            )

            energies = np.array(energies)
            magnetizations = np.array(magnetizations)

            # Calculate averages (per spin)
            E_avg = np.mean(energies) / self.N
            M_avg = np.mean(np.abs(magnetizations)) / self.N

            # Calculate fluctuations for specific heat and susceptibility
            E2_avg = np.mean(energies**2)
            M2_avg = np.mean(magnetizations**2)

            # Specific heat: C = (⟨E²⟩ - ⟨E⟩²) / (k_B T²)
            C = (E2_avg - np.mean(energies)**2) / (self.N * T**2)

            # Susceptibility: χ = (⟨M²⟩ - ⟨|M|⟩²) / (k_B T)
            chi = (M2_avg - np.mean(np.abs(magnetizations))**2) / (self.N * T)

            avg_energies.append(E_avg)
            avg_magnetizations.append(M_avg)
            specific_heats.append(C)
            susceptibilities.append(chi)
            energy_stds.append(np.std(energies) / self.N)
            magnetization_stds.append(np.std(np.abs(magnetizations)) / self.N)

        return SimulationResults(
            temperatures=temperatures,
            energies=np.array(avg_energies),
            magnetizations=np.array(avg_magnetizations),
            specific_heats=np.array(specific_heats),
            susceptibilities=np.array(susceptibilities),
            energy_stds=np.array(energy_stds),
            magnetization_stds=np.array(magnetization_stds)
        )

    def exact_energy_1d(self, temperature: float) -> float:
        """
        Calculate the exact energy per spin for 1D Ising model (h=0).

        For the 1D Ising model with periodic boundary conditions and h=0,
        the exact solution is: ⟨E⟩/N = -J * tanh(J/T)

        Args:
            temperature: Temperature in units where k_B = 1

        Returns:
            Exact energy per spin
        """
        if self.h != 0:
            raise ValueError("Exact solution only valid for h=0")
        return -self.J * np.tanh(self.J / temperature)

    def exact_specific_heat_1d(self, temperature: float) -> float:
        """
        Calculate the exact specific heat for 1D Ising model (h=0).

        Args:
            temperature: Temperature in units where k_B = 1

        Returns:
            Exact specific heat per spin
        """
        if self.h != 0:
            raise ValueError("Exact solution only valid for h=0")
        x = self.J / temperature
        return (x / np.cosh(x))**2


def plot_results(results: SimulationResults, model: IsingModel1D, save_path: Optional[str] = None):
    """
    Plot simulation results with comparison to exact solutions.

    Args:
        results: SimulationResults object from temperature_sweep
        model: IsingModel1D instance used for exact solutions
        save_path: If provided, save the figure to this path
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    T = results.temperatures

    # Energy per spin
    ax = axes[0, 0]
    ax.errorbar(T, results.energies, yerr=results.energy_stds,
                fmt='o', markersize=4, capsize=2, label='Monte Carlo')
    if model.h == 0:
        T_exact = np.linspace(T.min(), T.max(), 200)
        E_exact = [model.exact_energy_1d(t) for t in T_exact]
        ax.plot(T_exact, E_exact, 'r-', linewidth=2, label='Exact')
    ax.set_xlabel('Temperature (T)')
    ax.set_ylabel('Energy per spin (E/N)')
    ax.set_title('Energy vs Temperature')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Magnetization per spin
    ax = axes[0, 1]
    ax.errorbar(T, results.magnetizations, yerr=results.magnetization_stds,
                fmt='o', markersize=4, capsize=2, label='Monte Carlo')
    ax.set_xlabel('Temperature (T)')
    ax.set_ylabel('|Magnetization| per spin (|M|/N)')
    ax.set_title('Magnetization vs Temperature')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Specific heat
    ax = axes[1, 0]
    ax.plot(T, results.specific_heats, 'o', markersize=4, label='Monte Carlo')
    if model.h == 0:
        C_exact = [model.exact_specific_heat_1d(t) for t in T_exact]
        ax.plot(T_exact, C_exact, 'r-', linewidth=2, label='Exact')
    ax.set_xlabel('Temperature (T)')
    ax.set_ylabel('Specific Heat (C)')
    ax.set_title('Specific Heat vs Temperature')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Susceptibility
    ax = axes[1, 1]
    ax.plot(T, results.susceptibilities, 'o', markersize=4, label='Monte Carlo')
    ax.set_xlabel('Temperature (T)')
    ax.set_ylabel('Susceptibility (χ)')
    ax.set_title('Magnetic Susceptibility vs Temperature')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle(f'1D Ising Model: N={model.N}, J={model.J}, h={model.h}', fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    plt.show()


def plot_spin_evolution(model: IsingModel1D, temperature: float, n_steps: int = 100):
    """
    Visualize the time evolution of spins.

    Args:
        model: IsingModel1D instance
        temperature: Simulation temperature
        n_steps: Number of Monte Carlo sweeps to record
    """
    model.initialize_spins("random")
    beta = 1.0 / temperature

    spin_history = [model.spins.copy()]

    for _ in range(n_steps):
        model.monte_carlo_sweep(beta)
        spin_history.append(model.spins.copy())

    spin_history = np.array(spin_history)

    plt.figure(figsize=(12, 6))
    plt.imshow(spin_history.T, aspect='auto', cmap='coolwarm',
               interpolation='nearest', vmin=-1, vmax=1)
    plt.colorbar(label='Spin')
    plt.xlabel('Monte Carlo Sweep')
    plt.ylabel('Spin Site')
    plt.title(f'Spin Evolution at T={temperature}')
    plt.show()


def main():
    """Run example simulations."""
    print("=" * 60)
    print("1D Ising Model Monte Carlo Simulation")
    print("=" * 60)

    # Create model
    N = 100  # Number of spins
    J = 1.0  # Ferromagnetic coupling
    h = 0.0  # No external field

    model = IsingModel1D(N=N, J=J, h=h)

    print(f"\nModel parameters:")
    print(f"  N (chain length) = {N}")
    print(f"  J (coupling)     = {J}")
    print(f"  h (ext. field)   = {h}")

    # Run temperature sweep
    print("\nRunning temperature sweep...")
    results = model.temperature_sweep(
        T_min=0.5,
        T_max=5.0,
        n_temps=30,
        n_equilibrate=500,
        n_measure=2000,
        measure_interval=5,
        verbose=True
    )

    print("\nSimulation complete!")

    # Print some results
    print("\nSample results:")
    print("-" * 50)
    print(f"{'Temperature':>12} {'Energy/N':>12} {'|M|/N':>12} {'C':>12}")
    print("-" * 50)
    for i in range(0, len(results.temperatures), 5):
        print(f"{results.temperatures[i]:>12.3f} "
              f"{results.energies[i]:>12.4f} "
              f"{results.magnetizations[i]:>12.4f} "
              f"{results.specific_heats[i]:>12.4f}")

    # Compare with exact solution at T=1.0
    T_test = 1.0
    exact_E = model.exact_energy_1d(T_test)
    print(f"\nExact energy per spin at T={T_test}: {exact_E:.4f}")

    # Plot results
    print("\nGenerating plots...")
    plot_results(results, model)

    # Show spin evolution at different temperatures
    print("\nPlotting spin evolution at low temperature (T=0.5)...")
    model_viz = IsingModel1D(N=50, J=1.0, h=0.0)
    plot_spin_evolution(model_viz, temperature=0.5, n_steps=100)

    print("\nPlotting spin evolution at high temperature (T=3.0)...")
    plot_spin_evolution(model_viz, temperature=3.0, n_steps=100)


if __name__ == "__main__":
    main()
