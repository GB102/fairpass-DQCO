# Fork changes
This repository is a specialized fork that replaces standard Quantum Approximate Optimization Algorithm (QAOA) implementations with Digitized Counterdiabatic Quantum Optimization (DQCO) (also known as DC-QAOA).

By incorporating analytic counterdiabatic (CD) driving terms derived from the adiabatic gauge potential, this framework introduces "shortcuts to adiabaticity." While this increases circuit complexity and training overhead per iteration, it dramatically suppresses non-adiabatic transitions, yielding higher approximation ratios and superior scaling for larger problem window sizes.

## Key Enhancements: ##
- **Counterdiabatic Hamiltonian Integration ($H_{CD}$):** Expands the standard alternating ansatz by introducing first-order CD terms ($i[H_B, H_C]$), resulting in multi-body interaction strings (single-qubit $Y$ and two-qubit $YZ/ZY$ operations).
- **Three-Parameter Layer Structure:** Evolves the quantum state using a composite unitary combining Cost ($H_C$), Counterdiabatic ($H_{CD}$), and Transverse Mixer ($H_B$) drivers
- **Dual Simulation & Compilation Backends:** Includes both a high-performance NumpyDQCO statevector simulator and a Qiskit dqco_circuit builder equipped with automated basis-change gadgetizations.
