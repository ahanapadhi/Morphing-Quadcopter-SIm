# Morphing Quadcopter Simulation

A robotics simulation exploring how morphing quadcopter configurations can improve navigation in confined environments, with a focus on cave search-and-rescue scenarios.

The project combines procedural environment generation, simulated perception, occupancy-grid mapping, pose estimation, scan matching, and Kalman filtering to study autonomous navigation under sensor noise and localization uncertainty.

> **Status: Active development**

## Overview

Navigating narrow and irregular environments is challenging for aerial robots with fixed geometries. This project explores morphing as a form of robotic adaptation, allowing a quadcopter to change its physical configuration depending on its surroundings.

The simulation currently models three drone morphologies:

* **H** — compact configuration
* **O** — enclosed configuration
* **T** — elongated configuration

The drone navigates procedurally generated cave environments using simulated onboard sensors rather than direct access to the ground-truth map.

## Key Features

### Procedural Cave Generation

Cave environments are generated procedurally using randomized tunneling and corridor carving.

Environments can contain:

* Variable-width tunnels
* Turns and intersections
* Chambers
* Obstacles
* Irregular passageways

This allows the navigation system to be tested across different environments rather than a single fixed map.

### Morphing Drone Configurations

The simulation represents multiple drone morphologies:

```text
H        O        T
┌───┐   ╭───╮    ─────
│   │   │   │      │
└───┘   ╰───╯      │
```

Each morphology changes the drone's physical footprint and sensor placement, allowing their effects on navigation and maneuverability to be studied.

### Sensor Simulation

The drone uses six directional distance sensors positioned around its body.

Each sensor includes:

* Configurable sensing cone
* Field of view
* Minimum detection range
* Maximum detection range
* Measurement noise
* No-detection conditions

The navigation system receives only simulated sensor measurements rather than the environment's ground-truth geometry.

### Occupancy-Grid Mapping

The drone builds an occupancy-grid representation of its surroundings using:

**Sensor measurements + estimated pose → local map**

The mapper is intentionally restricted to information that would be available to the robot in a real environment.

### Pose Estimation

An IMU-style pose estimator models uncertainty and accumulated drift during dead reckoning.

The simulation includes:

* Position and heading estimation
* White noise
* Bias
* Bias random walk
* Dead-reckoning drift

This creates a distinction between the drone's true simulated pose and its estimated pose.

### Scan Matching

Scan matching compares successive sensor observations to estimate changes in the drone's pose and reduce accumulated localization error.

### Kalman Filtering

A diagonal Kalman filter combines pose estimates while weighting each dimension according to its uncertainty.

The filter estimates:

* X position
* Y position
* Heading

This allows noisy measurements and drifting dead reckoning to be combined into a more stable pose estimate.

## System Architecture

```text
             Procedural Cave
                    |
                    v
             Drone Morphology
                    |
                    v
              6x Distance Sensors
                    |
                    v
          Noisy Sensor Measurements
                    |
          +---------+---------+
          |                   |
          v                   v
   Occupancy Mapping     Pose Estimation
                              |
                    +---------+---------+
                    |                   |
                    v                   v
              Dead Reckoning      Scan Matching
                    |                   |
                    +---------+---------+
                              |
                              v
                       Kalman Filter
                              |
                              v
                    Estimated Drone Pose
```

## Research Questions

The simulation is designed to investigate:

1. How does drone morphology affect maneuverability in confined environments?
2. Can morphology changes improve navigation through narrow passages?
3. How does sensor noise affect environment reconstruction?
4. How quickly does localization error accumulate through dead reckoning?
5. How effectively can scan matching and filtering reduce localization drift?
6. Which morphology performs best across different cave geometries?

## Current Progress

### Implemented

* [x] Procedural cave generation
* [x] H, O, and T drone morphologies
* [x] Six-sensor distance model
* [x] Sensor noise and detection limits
* [x] Occupancy-grid mapping
* [x] IMU-style pose estimation
* [x] Dead-reckoning drift
* [x] Scan matching
* [x] Kalman filtering

### Planned

* [ ] Morphology transition simulation
* [ ] Autonomous morphology selection
* [ ] Path planning
* [ ] Navigation benchmarking
* [ ] Quantitative comparison between morphologies
* [ ] Expanded cave environments
* [ ] Performance visualization

## Why Morphing?

A fixed drone geometry forces the robot to navigate around its physical limitations.

A morphing drone introduces another degree of freedom:

> **Instead of only changing its path, the robot can change its shape.**

The long-term goal of this project is to explore whether morphology can become an active part of autonomous decision-making for aerial robots operating in constrained environments.

## Tech Stack

* Python
* Robotics Simulation
* Procedural Environment Generation
* Occupancy-Grid Mapping
* Sensor Modeling
* State Estimation
* Kalman Filtering
* Scan Matching

## Repository Structure

```text
morphing-quadcopter-sim/
|
├── src/
│   ├── cave/
│   ├── drone/
│   ├── sensors/
│   ├── mapping/
│   ├── localization/
│   └── simulation/
|
├── assets/
│   └── screenshots/
|
├── tests/
|
├── requirements.txt
└── README.md
```

## Getting Started

Clone the repository:

```bash
git clone https://github.com/<username>/morphing-quadcopter-sim.git
cd morphing-quadcopter-sim
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the simulation:

```bash
python <main_file>.py
```

Setup and execution instructions will be updated as the project develops.

## Author

**Ahana Padhi**

High school engineer and builder interested in robotics, embedded systems, mechanical design, autonomous systems, and human-centered engineering.
