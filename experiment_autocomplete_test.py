"""
Test file to demonstrate autocomplete functionality for experiments.

This file shows different ways to access experiment configurations
with proper type hints that enable IDE autocomplete.
"""

from rag_shared.utils.config import Config

def test_experiment_autocomplete():
    """Demonstrates different ways to access experiments with autocomplete."""
    
    # Initialize config
    cfg = Config(
        key_vault_name="RecoveredSpacesKV",
        config_folder="resources/configs",
        config_filename="handbook_config.yml"
    )
    
    print("=== Experiment Access Methods ===\n")
    
    # Method 1: Using the new get_experiment method (RECOMMENDED)
    # This provides full autocomplete for the Experiment object
    print("Method 1: Using cfg.get_experiment() - RECOMMENDED")
    experiment = cfg.get_experiment("prompt_optimization")
    if experiment:
        print(f"  Original enabled status: {experiment.enabled}")
        print(f"  Experiment name: {experiment.name}")
        print(f"  Status: {experiment.status}")
        print(f"  Traffic split: {experiment.traffic_split}")
        
        # You should get autocomplete for all these properties:
        # experiment.enabled
        # experiment.name  
        # experiment.status
        # experiment.traffic_split
        # experiment.variants
        # experiment.success_metrics
        
        experiment.enabled = True
        print(f"  After enabling: {experiment.enabled}")
    else:
        print("  Experiment 'prompt_optimization' not found")
    
    print("\nMethod 2: Using experiments config directly with __getitem__")
    # This also provides autocomplete now due to our __getitem__ method
    if cfg.app.experiments:
        try:
            exp = cfg.app.experiments["prompt_optimization"]  # Now has autocomplete!
            print(f"  Direct access - enabled: {exp.enabled}")
            print(f"  Direct access - name: {exp.name}")
            
            # You should get autocomplete here too:
            # exp.enabled
            # exp.name
            # exp.status, etc.
            
        except KeyError:
            print("  Experiment not found via direct access")
    
    print("\nMethod 3: Using get_experiment method from ExperimentsConfig")
    if cfg.app.experiments:
        exp = cfg.app.experiments.get_experiment("prompt_optimization")
        if exp:
            print(f"  Via experiments.get_experiment() - enabled: {exp.enabled}")
            # This also has full autocomplete support
        else:
            print("  Experiment not found via experiments.get_experiment()")

if __name__ == "__main__":
    test_experiment_autocomplete()
