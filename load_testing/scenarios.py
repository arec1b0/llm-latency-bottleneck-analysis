"""
Predefined Load Testing Scenarios

Ready-to-use test scenarios for different load patterns.
"""

from typing import Dict, List, Any


class LoadScenarios:
    """
    Collection of predefined load testing scenarios.
    
    Each scenario defines:
    - User count
    - Spawn rate
    - Run time
    - User class
    - Description
    """
    
    SCENARIOS: Dict[str, Dict[str, Any]] = {
        "baseline": {
            "users": 10,
            "spawn_rate": 2,
            "run_time": "5m",
            "user_class": "LLMInferenceUser",
            "description": "Baseline load - 10 concurrent users with gradual spawn",
            "purpose": "Establish baseline performance metrics",
            "expected_rps": "1-2 requests/sec",
        },
        
        "medium_load": {
            "users": 25,
            "spawn_rate": 5,
            "run_time": "10m",
            "user_class": "LLMInferenceUser",
            "description": "Medium load - 25 concurrent users",
            "purpose": "Test typical production load",
            "expected_rps": "3-5 requests/sec",
        },
        
        "high_load": {
            "users": 50,
            "spawn_rate": 10,
            "run_time": "10m",
            "user_class": "LLMInferenceUser",
            "description": "High load - 50 concurrent users",
            "purpose": "Identify performance degradation under stress",
            "expected_rps": "5-10 requests/sec",
        },
        
        "stress_test": {
            "users": 100,
            "spawn_rate": 20,
            "run_time": "15m",
            "user_class": "HighLoadUser",
            "description": "Stress test - 100 aggressive users",
            "purpose": "Find breaking point and OOM conditions",
            "expected_rps": "15-20 requests/sec",
        },
        
        "burst_traffic": {
            "users": 30,
            "spawn_rate": 30,
            "run_time": "10m",
            "user_class": "BurstLoadUser",
            "description": "Burst traffic - Sudden spikes in load",
            "purpose": "Test queue handling and recovery",
            "expected_rps": "Variable (0-15 requests/sec)",
        },
        
        "endurance": {
            "users": 20,
            "spawn_rate": 4,
            "run_time": "30m",
            "user_class": "LLMInferenceUser",
            "description": "Endurance test - Moderate load for extended period",
            "purpose": "Test memory leaks and long-term stability",
            "expected_rps": "2-3 requests/sec",
        },
        
        "quick_smoke": {
            "users": 5,
            "spawn_rate": 5,
            "run_time": "2m",
            "user_class": "LLMInferenceUser",
            "description": "Quick smoke test - Verify basic functionality",
            "purpose": "Fast sanity check before deeper testing",
            "expected_rps": "<1 request/sec",
        },
        
        "ramp_up": {
            "users": 50,
            "spawn_rate": 1,
            "run_time": "20m",
            "user_class": "LLMInferenceUser",
            "description": "Gradual ramp-up - Slow increase to 50 users",
            "purpose": "Observe degradation curve as load increases",
            "expected_rps": "Gradually increasing to 5-8 requests/sec",
        },
    }
    
    @classmethod
    def get_scenario(cls, name: str) -> Dict[str, Any]:
        """
        Get scenario by name.
        
        Args:
            name: Scenario name (e.g., 'baseline', 'medium_load')
        
        Returns:
            Scenario configuration dictionary
        
        Raises:
            KeyError: If scenario name not found
        """
        if name not in cls.SCENARIOS:
            available = ", ".join(cls.SCENARIOS.keys())
            raise KeyError(f"Scenario '{name}' not found. Available: {available}")
        
        return cls.SCENARIOS[name]
    
    @classmethod
    def list_scenarios(cls) -> List[str]:
        """Get list of available scenario names."""
        return list(cls.SCENARIOS.keys())
    
    @classmethod
    def print_scenario_details(cls, name: str) -> None:
        """
        Print detailed information about a scenario.
        
        Args:
            name: Scenario name
        """
        scenario = cls.get_scenario(name)
        
        print(f"\n{'=' * 60}")
        print(f"Scenario: {name}")
        print(f"{'=' * 60}")
        print(f"Description:  {scenario['description']}")
        print(f"Purpose:      {scenario['purpose']}")
        print(f"\nConfiguration:")
        print(f"  Users:       {scenario['users']}")
        print(f"  Spawn Rate:  {scenario['spawn_rate']} users/sec")
        print(f"  Run Time:    {scenario['run_time']}")
        print(f"  User Class:  {scenario['user_class']}")
        print(f"\nExpected Load:")
        print(f"  {scenario['expected_rps']}")
        print(f"{'=' * 60}\n")
    
    @classmethod
    def print_all_scenarios(cls) -> None:
        """Print information about all scenarios."""
        print("\n" + "=" * 60)
        print("AVAILABLE LOAD TESTING SCENARIOS")
        print("=" * 60 + "\n")
        
        for name in cls.list_scenarios():
            scenario = cls.SCENARIOS[name]
            print(f"📊 {name.upper()}")
            print(f"   {scenario['description']}")
            print(f"   Users: {scenario['users']}, "
                  f"Duration: {scenario['run_time']}, "
                  f"Rate: {scenario['spawn_rate']}/sec")
            print()


def generate_command(scenario_name: str, host: str = "http://localhost:8000") -> str:
    """
    Generate Locust command line for a scenario.
    
    Args:
        scenario_name: Name of the scenario
        host: API host URL
    
    Returns:
        Complete Locust command
    """
    scenario = LoadScenarios.get_scenario(scenario_name)
    
    command = (
        f"locust -f locustfile.py "
        f"--users {scenario['users']} "
        f"--spawn-rate {scenario['spawn_rate']} "
        f"--run-time {scenario['run_time']} "
        f"--host {host}"
    )
    
    # Add user class if specified
    if scenario.get("user_class") and scenario["user_class"] != "LLMInferenceUser":
        command += f" --user-class {scenario['user_class']}"
    
    # Add headless mode for automated testing
    command += " --headless"
    
    return command


def get_recommended_sequence() -> List[str]:
    """
    Get recommended sequence of scenarios for comprehensive testing.
    
    Returns:
        Ordered list of scenario names
    """
    return [
        "quick_smoke",     # 1. Verify basic functionality
        "baseline",        # 2. Establish baseline
        "medium_load",     # 3. Test typical load
        "ramp_up",         # 4. Observe degradation curve
        "high_load",       # 5. Test under pressure
        "burst_traffic",   # 6. Test spike handling
        "stress_test",     # 7. Find breaking point
        "endurance",       # 8. Test long-term stability (optional)
    ]


def print_recommended_workflow() -> None:
    """Print recommended testing workflow."""
    sequence = get_recommended_sequence()
    
    print("\n" + "=" * 60)
    print("RECOMMENDED TESTING WORKFLOW")
    print("=" * 60 + "\n")
    
    for i, scenario_name in enumerate(sequence, 1):
        scenario = LoadScenarios.get_scenario(scenario_name)
        print(f"{i}. {scenario_name.upper()}")
        print(f"   Purpose: {scenario['purpose']}")
        print(f"   Command: {generate_command(scenario_name)}")
        print()
    
    print("Tip: Run scenarios in order, analyzing results between runs")
    print("=" * 60 + "\n")


# Example usage
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        scenario_name = sys.argv[1]
        
        if scenario_name == "--list":
            LoadScenarios.print_all_scenarios()
        elif scenario_name == "--workflow":
            print_recommended_workflow()
        else:
            try:
                LoadScenarios.print_scenario_details(scenario_name)
                print("\nCommand to run:")
                print(generate_command(scenario_name))
            except KeyError as e:
                print(f"Error: {e}")
                print("\nUse --list to see available scenarios")
    else:
        print("Usage:")
        print("  python scenarios.py --list                # List all scenarios")
        print("  python scenarios.py --workflow            # Show recommended workflow")
        print("  python scenarios.py <scenario_name>       # Show scenario details")
        print("\nExamples:")
        print("  python scenarios.py baseline")
        print("  python scenarios.py stress_test")
