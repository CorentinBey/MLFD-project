# MLFD-project

# Trajectory Generation and Drone Dynamics Toolkit

## 📋 Overview

This module provides a full set of tools to:

- Generate **smooth 3D drone trajectories** between waypoints using polynomial optimization.
- Calculate **desired and actual drone motion** (position, velocity, acceleration).
- Compute **optimal thrust** and **attitude profiles** to follow the desired accelerations.
- Model **body rates**, **moments**, and **individual propeller thrusts**.
- Add **noise models** and **apply filtering** (Butterworth + LOWESS smoothing).
- Predict **propeller RPM** and **torques** using a **Neural Network** model (BEMT\_NN).

It is highly flexible for **trajectory tracking simulations**, **drone controller design**, and **machine learning-based performance analysis**.

---

## 📂 Structure

| File/Class   | Description                                                                 |
| ------------ | --------------------------------------------------------------------------- |
| `Trajectory` | Main class to generate trajectories and all physical quantities.            |
| `Drone`      | Base class containing physical parameters (mass, inertia, arm length).      |
| `BEMT_NN`    | Neural Network model to predict RPM and torque from aerodynamic conditions. |
| `Data_rpm`   | (Separate) Class for RPM data handling, filtering, and plotting.            |

---

## ⚙️ How It Works

1. **Initialization**:

   ```python
   traj = Trajectory(x_waypoints, y_waypoints, z_waypoints, t_waypoints, key, total_steps, drone_object)
   ```

2. **Trajectory Generation** (Automatic inside constructor):

   - Polynomial optimization using boundary and continuity conditions.
   - Full 3D desired trajectories computed.
   - Optimal thrust and attitude trajectories computed.

3. **Post-processing Tools**:

   - Body rates and moments computed.
   - Individual thrust per propeller calculated.
   - RPM and torque predictions from BEMT-based NN.

4. **Optional**:

   - Add noise to thrust and RPM for simulation realism.
   - Apply Butterworth and/or LOWESS smoothing.
   - Generate multiple plots (3D path, time profiles of velocities, accelerations, RPMs).

---

## 🧹 Main Functions

| Function                                    | Purpose                                                      |
| ------------------------------------------- | ------------------------------------------------------------ |
| `_trajectory_algorithm`                     | Solve for trajectory polynomial coefficients.                |
| `_T_attitude_optimal`                       | Optimize thrust and attitude to match desired accelerations. |
| `_get_bodyrates`                            | Compute body angular rates from Euler angles.                |
| `_compute_individual_Thrust_prop`           | Compute per-propeller thrusts from total thrust and moments. |
| `_get_rpm_and_Q_net_from_NN`                | Predict RPM and torques using Neural Network model.          |
| `add_noise_thrust`, `add_noise_rpm`         | Add noise to simulate measurement uncertainties.             |
| `filter_noise`, `_filter_LOWESS_method_vec` | Smooth signals.                                              |
| Plotting functions                          | Visualize trajectories, RPM, thrust, body rates, etc.        |

---

## 🛠️ Requirements

- **Python 3.8+**
- Libraries:
  - `numpy`
  - `scipy`
  - `matplotlib`
  - `pandas`
  - `statsmodels`
  - `sympy`
  - `scikit-learn` (for Neural Network, if needed)

Install with:

```bash
pip install numpy scipy matplotlib pandas statsmodels sympy scikit-learn
```

---

## 🚀 Quick Example

```python
# Define waypoints
x = [0, 5, 10]
y = [0, 5, 0]
z = [0, 2, 0]
t = [0, 5, 10]

# Create drone object with physical parameters
drone = Drone(mass=1.5, g=9.81, I=np.eye(3), l_arm=0.25)

# Generate trajectory
traj = Trajectory(x, y, z, t, key="example", total_steps=1000, drone=drone)

# Plot desired 3D trajectory
traj.plot_trajectory(data=some_measured_data)

# Get desired positions
positions = traj.get_desired_positions()
```

---

## 📊 Outputs Available

- Desired and actual **positions**, **velocities**, **accelerations**.
- **Optimal thrust** and **attitude angles** (roll, pitch).
- **RPM predictions** for each propeller.
- **Body rates** and **moments**.
- **Plots** for all key quantities.

---

## 📌 Notes

- The system supports **4th-order polynomial optimization** for **snap minimization** (smoothest motion).
- Customizable **noise models** and **filtering methods**.
- Extendable for **neural network re-training** if needed with new data.


