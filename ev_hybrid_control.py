"""
Hybrid Neural Network and Fuzzy Logic Control System for Electric Vehicle Performance Optimization

This implementation demonstrates the integration of neural networks and fuzzy logic
for optimizing electric vehicle performance, including energy management, motor control,
and battery management.

Author: Satbir Singh
Paper: A Review of Hybrid Neural Network and Fuzzy Logic Control Techniques 
       for Optimising Electric Vehicle Performance
"""

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, Dict, List
import math


class FuzzyLogicController:
    """
    Fuzzy Logic Controller for EV motor control and energy management.
    Implements rule-based decision making for real-time control.
    """
    
    def __init__(self):
        self.membership_functions = self._initialize_membership_functions()
    
    def _initialize_membership_functions(self) -> Dict:
        """Initialize membership functions for fuzzy variables"""
        return {
            'speed': {
                'low': lambda x: max(0, 1 - x/30) if x <= 30 else 0,
                'medium': lambda x: max(0, min((x-20)/20, (60-x)/20)) if 20 <= x <= 60 else 0,
                'high': lambda x: max(0, (x-50)/30) if x >= 50 else 0
            },
            'battery_soc': {
                'low': lambda x: max(0, 1 - x/30) if x <= 30 else 0,
                'medium': lambda x: max(0, min((x-40)/20, (80-x)/20)) if 40 <= x <= 80 else 0,
                'high': lambda x: max(0, (x-70)/30) if x >= 70 else 0
            },
            'torque_demand': {
                'low': lambda x: max(0, 1 - x/50) if x <= 50 else 0,
                'medium': lambda x: max(0, min((x-40)/30, (100-x)/30)) if 40 <= x <= 100 else 0,
                'high': lambda x: max(0, (x-80)/50) if x >= 80 else 0
            }
        }
    
    def fuzzify(self, variable: str, value: float) -> Dict[str, float]:
        """Convert crisp input to fuzzy membership values"""
        if variable not in self.membership_functions:
            return {}
        
        memberships = {}
        for term, func in self.membership_functions[variable].items():
            memberships[term] = func(value)
        
        return memberships
    
    def infer(self, speed: float, battery_soc: float, torque_demand: float) -> float:
        """
        Fuzzy inference engine for motor torque control
        Implements rule-based decision making
        """
        speed_fuzzy = self.fuzzify('speed', speed)
        soc_fuzzy = self.fuzzify('battery_soc', battery_soc)
        torque_fuzzy = self.fuzzify('torque_demand', torque_demand)
        
        # Rule base: IF-THEN rules for torque control
        rules = []
        
        # Rule 1: IF speed is low AND SOC is high THEN torque is high
        rule1 = min(speed_fuzzy.get('low', 0), soc_fuzzy.get('high', 0))
        rules.append(('high', rule1))
        
        # Rule 2: IF speed is medium AND SOC is medium THEN torque is medium
        rule2 = min(speed_fuzzy.get('medium', 0), soc_fuzzy.get('medium', 0))
        rules.append(('medium', rule2))
        
        # Rule 3: IF speed is high OR SOC is low THEN torque is low
        rule3 = max(speed_fuzzy.get('high', 0), soc_fuzzy.get('low', 0))
        rules.append(('low', rule3))
        
        # Defuzzification using centroid method
        output_torque = self._defuzzify(rules)
        return output_torque
    
    def _defuzzify(self, rules: List[Tuple[str, float]]) -> float:
        """Defuzzify fuzzy output to crisp value using centroid method"""
        if not rules:
            return 0.0
        
        # Simplified centroid defuzzification
        total_weight = sum(weight for _, weight in rules)
        if total_weight == 0:
            return 0.0
        
        # Map linguistic terms to numeric values
        term_values = {'low': 30, 'medium': 60, 'high': 90}
        
        weighted_sum = sum(term_values.get(term, 0) * weight for term, weight in rules)
        return weighted_sum / total_weight if total_weight > 0 else 0.0


class NeuralNetworkPredictor:
    """
    Neural Network for predictive control and energy optimization in EVs.
    Uses deep learning for pattern recognition and future state prediction.
    """
    
    def __init__(self, input_dim: int = 5, hidden_layers: List[int] = [64, 32, 16]):
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.model = self._build_model()
        self.scaler = MinMaxScaler()
        self.is_trained = False
    
    def _build_model(self) -> tf.keras.Model:
        """Build neural network model for energy prediction"""
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(self.hidden_layers[0], activation='relu', 
                                input_shape=(self.input_dim,)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(self.hidden_layers[1], activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(self.hidden_layers[2], activation='relu'),
            tf.keras.layers.Dense(1, activation='linear')  # Energy consumption prediction
        ])
        
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              epochs: int = 100, batch_size: int = 32):
        """Train the neural network on historical driving data"""
        # Normalize features
        X_scaled = self.scaler.fit_transform(X_train)
        
        # Train model
        history = self.model.fit(
            X_scaled, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            verbose=0
        )
        
        self.is_trained = True
        return history
    
    def predict_energy_consumption(self, features: np.ndarray) -> float:
        """
        Predict energy consumption based on current vehicle state
        Features: [speed, acceleration, road_grade, temperature, battery_soc]
        """
        if not self.is_trained:
            # Return default prediction if not trained
            return 0.5
        
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        prediction = self.model.predict(features_scaled, verbose=0)[0][0]
        return max(0.0, prediction)  # Ensure non-negative


class HybridEVController:
    """
    Hybrid control system combining Neural Networks and Fuzzy Logic
    for optimal EV performance management.
    """
    
    def __init__(self):
        self.fuzzy_controller = FuzzyLogicController()
        self.nn_predictor = NeuralNetworkPredictor()
        self.energy_history = []
        self.performance_metrics = {
            'total_energy_consumed': 0.0,
            'average_efficiency': 0.0,
            'battery_cycles': 0
        }
    
    def optimize_energy_management(self, vehicle_state: Dict) -> Dict:
        """
        Optimize energy management using hybrid approach
        Combines NN prediction with fuzzy logic control
        """
        speed = vehicle_state.get('speed', 0)
        battery_soc = vehicle_state.get('battery_soc', 50)
        torque_demand = vehicle_state.get('torque_demand', 0)
        acceleration = vehicle_state.get('acceleration', 0)
        road_grade = vehicle_state.get('road_grade', 0)
        temperature = vehicle_state.get('temperature', 25)
        
        # Neural network prediction for energy consumption
        nn_features = np.array([speed, acceleration, road_grade, temperature, battery_soc])
        predicted_energy = self.nn_predictor.predict_energy_consumption(nn_features)
        
        # Fuzzy logic control for motor torque
        optimal_torque = self.fuzzy_controller.infer(speed, battery_soc, torque_demand)
        
        # Hybrid decision: Combine predictions
        energy_allocation = self._calculate_energy_allocation(
            predicted_energy, optimal_torque, battery_soc
        )
        
        # Update performance metrics
        self._update_metrics(energy_allocation)
        
        return {
            'optimal_torque': optimal_torque,
            'predicted_energy': predicted_energy,
            'energy_allocation': energy_allocation,
            'motor_power': optimal_torque * speed * 0.001,  # Simplified power calculation
            'regenerative_braking': self._calculate_regenerative_braking(speed, battery_soc)
        }
    
    def _calculate_energy_allocation(self, predicted_energy: float, 
                                     torque: float, soc: float) -> Dict:
        """Calculate optimal energy allocation across subsystems"""
        total_available = soc * 0.01  # Convert SOC to energy units
        
        # Prioritize motor power
        motor_allocation = min(torque * 0.1, total_available * 0.7)
        
        # Allocate to auxiliary systems
        aux_allocation = total_available * 0.2
        
        # Reserve for battery management
        battery_reserve = total_available * 0.1
        
        return {
            'motor': motor_allocation,
            'auxiliary': aux_allocation,
            'battery_management': battery_reserve,
            'total': motor_allocation + aux_allocation + battery_reserve
        }
    
    def _calculate_regenerative_braking(self, speed: float, soc: float) -> float:
        """Calculate regenerative braking efficiency"""
        if speed < 10 or soc > 90:
            return 0.0  # No regeneration at low speed or high SOC
        
        # Regeneration efficiency increases with speed and decreases with SOC
        efficiency = min(0.8, speed / 50) * (1 - soc / 100)
        return max(0.0, efficiency)
    
    def _update_metrics(self, energy_allocation: Dict):
        """Update performance tracking metrics"""
        self.performance_metrics['total_energy_consumed'] += energy_allocation.get('total', 0)
        self.energy_history.append(energy_allocation.get('total', 0))
        
        if len(self.energy_history) > 100:
            self.energy_history.pop(0)
        
        if self.energy_history:
            self.performance_metrics['average_efficiency'] = np.mean(self.energy_history)
    
    def get_performance_report(self) -> Dict:
        """Get current performance metrics"""
        return {
            **self.performance_metrics,
            'energy_history_length': len(self.energy_history),
            'recent_efficiency': np.mean(self.energy_history[-10:]) if len(self.energy_history) >= 10 else 0
        }


def generate_training_data(n_samples: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic training data for neural network"""
    np.random.seed(42)
    
    # Features: [speed, acceleration, road_grade, temperature, battery_soc]
    X = np.random.rand(n_samples, 5)
    X[:, 0] *= 120  # Speed: 0-120 km/h
    X[:, 1] = (X[:, 1] - 0.5) * 4  # Acceleration: -2 to 2 m/s²
    X[:, 2] = (X[:, 2] - 0.5) * 10  # Road grade: -5 to 5 degrees
    X[:, 3] = X[:, 3] * 40 + 10  # Temperature: 10-50°C
    X[:, 4] *= 100  # Battery SOC: 0-100%
    
    # Energy consumption (simplified model)
    y = (X[:, 0] * 0.1 + abs(X[:, 1]) * 2 + abs(X[:, 2]) * 0.5 + 
         (X[:, 3] - 25) * 0.01 + (100 - X[:, 4]) * 0.05) + np.random.rand(n_samples) * 5
    
    return X, y


def main():
    """Demonstration of hybrid EV control system"""
    print("=" * 60)
    print("Hybrid Neural Network and Fuzzy Logic Control System")
    print("Electric Vehicle Performance Optimization")
    print("=" * 60)
    print()
    
    # Initialize hybrid controller
    controller = HybridEVController()
    
    # Train neural network
    print("Training Neural Network on historical driving data...")
    X_train, y_train = generate_training_data(1000)
    controller.nn_predictor.train(X_train, y_train, epochs=50)
    print("✓ Neural Network trained successfully")
    print()
    
    # Simulate vehicle operation
    print("Simulating EV operation with hybrid control...")
    print("-" * 60)
    
    test_scenarios = [
        {'speed': 30, 'battery_soc': 80, 'torque_demand': 50, 
         'acceleration': 1.0, 'road_grade': 0, 'temperature': 25},
        {'speed': 60, 'battery_soc': 50, 'torque_demand': 70, 
         'acceleration': 0.5, 'road_grade': 2, 'temperature': 30},
        {'speed': 90, 'battery_soc': 30, 'torque_demand': 80, 
         'acceleration': -1.0, 'road_grade': -1, 'temperature': 20},
    ]
    
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\nScenario {i}:")
        print(f"  Speed: {scenario['speed']} km/h, SOC: {scenario['battery_soc']}%")
        
        result = controller.optimize_energy_management(scenario)
        
        print(f"  Optimal Torque: {result['optimal_torque']:.2f} Nm")
        print(f"  Predicted Energy: {result['predicted_energy']:.2f} kWh")
        print(f"  Motor Power: {result['motor_power']:.2f} kW")
        print(f"  Regenerative Braking Efficiency: {result['regenerative_braking']:.2%}")
        print(f"  Energy Allocation:")
        for component, value in result['energy_allocation'].items():
            print(f"    - {component.capitalize()}: {value:.2f} kWh")
    
    print("\n" + "=" * 60)
    print("Performance Report:")
    print("=" * 60)
    report = controller.get_performance_report()
    for key, value in report.items():
        print(f"  {key.replace('_', ' ').title()}: {value:.2f}")
    print()


if __name__ == "__main__":
    main()

